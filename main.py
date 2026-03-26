from torch.utils.data import WeightedRandomSampler
from intent_pretrain import ExternalPretrainModelManager
from model import CLBert, BertForModel
from init_parameter import init_model
from dataloader import Data
from mtp import InternalPretrainModelManager
from utils.tools import *
from utils.memory import MemoryBank, fill_memory_bank
from utils.neighbor_dataset import NeighborsDataset
from utils.build_ml import set_args
from methods import *
import matplotlib.pyplot as plt
import umap
import time
from transformers import AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SADAModelManager:
    """
    The implementation of Contrastive Learning with Nearest Neighbors
    """
    def __init__(self, args, data, pretrained_model=None, student_model=None):
        set_seed(args.seed)
        n_gpu = torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = data.num_labels
        self.model = CLBert(args.bert_model, device=self.device, num_labels=self.num_labels)

        if n_gpu > 1:
            self.model = nn.DataParallel(self.model)
        
        if not args.disable_pretrain:
            self.pretrained_model = pretrained_model
            self.load_pretrained_model()
        
        self.num_train_optimization_steps = int(len(data.train_semi_dataset) / args.train_batch_size) * args.num_train_epochs
        self.student_model = student_model
        self.optimizer, self.scheduler = self.get_optimizer(args)
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.generator = view_generator(self.tokenizer, args.rtr_prob, args.seed)

    def get_neighbor_dataset(self, args, data, indices):
        """convert indices to dataset"""
        dataset = NeighborsDataset(data.train_semi_dataset, indices)
        all_labels = data.train_semi_dataset.tensors[3].numpy()
        labeled_mask = (all_labels != -1)
        unlabeled_mask = (all_labels == -1)
        N_labeled = labeled_mask.sum()
        if N_labeled > 0:
            known_labels = all_labels[labeled_mask]
            class_counts = np.bincount(known_labels, minlength=data.n_known_cls)
            class_counts = class_counts + 1e-6
            n_classes = len(class_counts)
            class_weights = N_labeled / (n_classes * class_counts)
            weights = np.zeros_like(all_labels, dtype=np.float32)
            weights[labeled_mask] = class_weights[all_labels[labeled_mask]]
            weights[unlabeled_mask] = 1.0
            weights = torch.FloatTensor(weights)
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            self.train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, sampler=sampler, shuffle=False, num_workers=4, pin_memory=True)
        else:
            self.train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    def get_neighbor_inds(self, args, data):
        """get indices of neighbors"""
        memory_bank = MemoryBank(len(data.train_semi_dataset), args.feat_dim, len(data.all_label_list), 0.1)
        fill_memory_bank(data.train_semi_dataloader, self.model, memory_bank)
        indices = memory_bank.mine_nearest_neighbors(args.topk, calculate_accuracy=False)
        return indices
    
    def get_adjacency(self, args, inds, neighbors, targets):
        """get adjacency matrix"""
        adj = torch.zeros(inds.shape[0], inds.shape[0])
        for b1, n in enumerate(neighbors):
            adj[b1][b1] = 1
            for b2, j in enumerate(inds):
                if j in n:
                    adj[b1][b2] = 1
                if (targets[b1] == targets[b2]) and (targets[b1]>0) and (targets[b2]>0):
                    adj[b1][b2] = 1
        return adj

    def evaluation(self, args, data, save_results=True, plot_cm=True):
        """final clustering evaluation on test set"""
        # get features
        feats_test, labels = self.get_features_labels(data.test_dataloader, self.model, args)
        feats_test = feats_test.cpu().numpy()

        # k-means clustering
        km = KMeans(n_clusters = self.num_labels).fit(feats_test)
        
        y_pred = km.labels_
        y_true = labels.cpu().numpy()

        results = clustering_score(y_true, y_pred)
        print('results',results)
        
        # confusion matrix
        if plot_cm:
            ind, _ = hungray_aligment(y_true, y_pred)
            map_ = {i[0]:i[1] for i in ind}
            y_pred = np.array([map_[idx] for idx in y_pred])

            cm = confusion_matrix(y_true,y_pred)   
            print('confusion matrix',cm)
            self.test_results = results
        
        # save results
        if save_results:
            self.save_results(args)

    def train(self, args, data):
        if isinstance(self.model, nn.DataParallel):
            criterion = self.model.module.loss_cl
        else:
            criterion = self.model.loss_cl
        
        # load neighbors for the first epoch
        indices = self.get_neighbor_inds(args, data)
        self.get_neighbor_dataset(args, data, indices)

        scaler = torch.cuda.amp.GradScaler()

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            if self.student_model:
                self.student_model.eval()
            for batch in tqdm(self.train_dataloader, desc="Iteration"):
                # 1. load data
                anchor = tuple(t.to(self.device) for t in batch["anchor"])
                neighbor = tuple(t.to(self.device) for t in batch["neighbor"])
                pos_neighbors = batch["possible_neighbors"]  # all possible neighbor inds for anchor
                data_inds = batch["index"]

                # 2. get adjacency matrix
                adjacency = self.get_adjacency(args, data_inds, pos_neighbors, batch["target"])  # (bz,bz)
                if args.view_strategy == "rtr":
                    X_an = {"input_ids": self.generator.random_token_replace(anchor[0].cpu()).to(self.device),
                            "attention_mask": anchor[1], "token_type_ids": anchor[2]}
                    X_ng = {"input_ids": self.generator.random_token_replace(neighbor[0].cpu()).to(self.device),
                            "attention_mask": neighbor[1], "token_type_ids": neighbor[2]}
                elif args.view_strategy == "shuffle":
                    X_an = {"input_ids": self.generator.shuffle_tokens(anchor[0].cpu()).to(self.device),
                            "attention_mask": anchor[1], "token_type_ids": anchor[2]}
                    X_ng = {"input_ids": self.generator.shuffle_tokens(neighbor[0].cpu()).to(self.device),
                            "attention_mask": neighbor[1], "token_type_ids": neighbor[2]}
                elif args.view_strategy == "none":
                    X_an = {"input_ids": anchor[0], "attention_mask": anchor[1], "token_type_ids": anchor[2]}
                    X_ng = {"input_ids": neighbor[0], "attention_mask": neighbor[1], "token_type_ids": neighbor[2]}

                elif args.view_strategy == "SADA":
                    if not self.student_model:
                        raise ValueError("Student model not found!")

                    def apply_student(input_ids, attention_mask, special_tokens_mask, do_viz=False):
                        if isinstance(self.model, nn.DataParallel):
                            backbone = self.model.module.backbone
                        else:
                            backbone = self.model.backbone
                        # raw_embeds = backbone.embeddings(input_ids=input_ids)
                        with torch.no_grad():
                            base_model = getattr(backbone, backbone.base_model_prefix, backbone)
                            raw_embeds = base_model.embeddings.word_embeddings(input_ids)

                            mask_prob = self.student_model(
                                raw_embeds,
                                attention_mask=attention_mask,
                                special_tokens_mask=special_tokens_mask,
                                temperature=0.15,
                                mask_threshold=args.mask_threshold
                            )
                            mu = torch.mean(raw_embeds, dim=-1, keepdim=True)
                            sigma = torch.std(raw_embeds, dim=-1, keepdim=True)
                            noise = torch.randn_like(raw_embeds) * sigma + mu
                        perturbed_embeds = (1 - mask_prob) * raw_embeds + mask_prob * noise

                        if do_viz:
                            print(f"\n[Student Augmentation Visualization - Epoch {epoch + 1}]")
                            input_id_sample = input_ids[0].cpu().numpy()
                            score_sample = mask_prob[0].squeeze(-1).cpu().detach().numpy()
                            tokens = self.tokenizer.convert_ids_to_tokens(input_id_sample)

                            print(f"{'Token':<15} | {'Score (Imp)':<12} | {'Action'}")
                            print("-" * 45)
                            for t, s in zip(tokens, score_sample):
                                if t == '[PAD]': continue
                                action = "MASK" if s > 1e-9 else "KEEP"
                                print(f"{t:<15} | {s:.4f}       | {action}")
                            print("-" * 45 + "\n")
                        return perturbed_embeds

                    visualize_now = (nb_tr_steps == 0) and (epoch < 5)
                    anchor_embeds = apply_student(anchor[0], anchor[1], anchor[4], do_viz=visualize_now)
                    X_an = {"inputs_embeds": anchor_embeds, "attention_mask": anchor[1], "token_type_ids": anchor[2]}
                    neighbor_embeds = apply_student(neighbor[0], neighbor[1], neighbor[4], do_viz=False)
                    X_ng = {"inputs_embeds": neighbor_embeds, "attention_mask": neighbor[1], "token_type_ids": neighbor[2]}
                else:
                    raise NotImplementedError(f"View strategy {args.view_strategy} not implemented!")

                # 4. compute loss and update parameters
                with torch.set_grad_enabled(True):
                    with torch.cuda.amp.autocast():
                        f_pos = torch.stack([self.model(X_an)["features"], self.model(X_ng)["features"]], dim=1)
                        loss = criterion(f_pos, mask=adjacency, temperature=args.temp)
                        tr_loss += loss.item()

                    # loss.backward()
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)

                    nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)

                    # self.optimizer.step()
                    scaler.step(self.optimizer)
                    scaler.update()

                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    nb_tr_examples += anchor[0].size(0)
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            print('train_loss',loss)

            # update neighbors every several epochs
            if ((epoch + 1) % args.update_per_epoch) == 0:
                indices = self.get_neighbor_inds(args, data)
                self.get_neighbor_dataset(args, data, indices)

    def get_optimizer(self, args):
        num_warmup_steps = int(args.warmup_proportion*self.num_train_optimization_steps)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=self.num_train_optimization_steps)
        return optimizer, scheduler
    
    def load_pretrained_model(self):
        """load the backbone of pretrained model"""
        if isinstance(self.pretrained_model, nn.DataParallel):
            pretrained_dict = self.pretrained_model.module.backbone.state_dict()
        else:
            pretrained_dict = self.pretrained_model.backbone.state_dict()
        if isinstance(self.model, nn.DataParallel):
            self.model.module.backbone.load_state_dict(pretrained_dict, strict=False)
        else:
            self.model.backbone.load_state_dict(pretrained_dict, strict=False)

    def get_features_labels(self, dataloader, model, args):
        model.eval()
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="Extracting representation"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch[:4]
            X = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
            with torch.no_grad():
                feature = model(X, output_hidden_states=True)["hidden_states"]

            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))

        return total_features, total_labels
            
    def save_results(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        # var = [args.internal_dataset, args.method, args.known_cls_ratio, args.labeled_ratio, args.topk, args.view_strategy, args.seed]
        # names = ['dataset', 'method', 'known_cls_ratio', 'labeled_ratio', 'topk', 'view_strategy', 'seed']
        var = [args.internal_dataset, args.known_cls_ratio, args.labeled_ratio, args.view_strategy, args.input_strategy,
                                   args.with_speaker, args.ratio, args.mask_threshold, args.seed]
        names = ['dataset', 'known_cls_ratio', 'labeled_ratio', 'view_strategy', 'input_strategy', 'with_speaker',
                             'teacher_mask_threshold', 'student_mask_threshold', 'seed']

        vars_dict = {k:v for k,v in zip(names, var)}
        results = dict(self.test_results,**vars_dict)
        keys = list(results.keys())
        values = list(results.values())
        
        file_name = 'results.csv'
        results_path = os.path.join(args.save_results_path, file_name)
        
        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori,columns = keys)
            df1.to_csv(results_path,index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results,index=[1])
            df1 = df1.append(new,ignore_index=True)
            df1.to_csv(results_path,index=False)
        data_diagram = pd.read_csv(results_path)
        
        print('test_results', data_diagram)

    def visualize_Tsne(self, data, args, n_samples=2000):
        """
        Visualize the learned embeddings of the final model on Test Set.
        Unlike the distillation step, here we use CLEAN data (no mask) to check cluster separation.
        """
        print("Visualizing embeddings...")
        self.model.eval()

        # 1. Collect Embeddings and Labels
        # We use the test_dataloader to see generalization
        dataloader = data.test_dataloader

        sample_representations = []
        labels = []

        # Limit samples to avoid overcrowding the plot
        count = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting embeddings for visualization"):
                # Move batch to device
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch[:4]

                # Construct input dictionary
                X = {
                    "input_ids": input_ids,
                    "attention_mask": input_mask,
                    "token_type_ids": segment_ids
                }

                # Forward pass
                # We want the backbone output (usually [CLS] token representation)
                # CLBert returns 'hidden_states' if output_hidden_states=True
                outputs = self.model(X, output_hidden_states=True)
                cls_embeddings = outputs["hidden_states"].detach().cpu().numpy()
                batch_labels = label_ids.detach().cpu().numpy()

                if len(sample_representations) == 0:
                    sample_representations = cls_embeddings
                    labels = batch_labels
                else:
                    sample_representations = np.concatenate([sample_representations, cls_embeddings])
                    labels = np.concatenate([labels, batch_labels])

                count += input_ids.size(0)
                if count >= n_samples:
                    break


        print(f"Collected {len(labels)} samples. Running UMAP...")

        # 2. Run UMAP (Dimensionality Reduction)
        reducer = umap.UMAP(random_state=args.seed)
        t0 = time.time()
        embedding_2d = reducer.fit_transform(sample_representations)
        print(f"UMAP finished in {time.time() - t0:.2f}s")

        # 3. Plotting
        title = f'UMAP projection of {args.internal_dataset} (Method: {args.method})'
        fig = self.plot_embedding(embedding_2d, labels, title)

        # Save the figure
        if args.save_results_path:
            file_name = f"vis_{args.internal_dataset}_K{args.known_cls_ratio}_L{args.labeled_ratio}_{args.view_strategy}_{args.input_strategy}_SP{args.with_speaker}_TE{args.ratio}_ST{args.mask_threshold}_S{args.seed}.png"
            save_path = os.path.join(args.save_results_path, file_name)
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")

        # Optional: Show plot if in an interactive environment
        # plt.show(fig)
        plt.close(fig)  # Close to free memory

    @staticmethod
    def plot_embedding(data, label, title):
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)

        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(111)

        # Define colors (ensure enough colors for classes)
        # Using a distinct colormap
        unique_labels = np.unique(label)
        n_classes = len(unique_labels)
        cmap = plt.cm.get_cmap('jet', n_classes)

        for i in range(data.shape[0]):
            plt.text(data[i, 0], data[i, 1], str(label[i]),
                     color=cmap(label[i] / n_classes),
                     fontdict={'weight': 'bold', 'size': 9})

        plt.xticks([])
        plt.yticks([])
        plt.title(title)
        return fig


if __name__ == '__main__':
    print('[Init] Data and Parameters Initialization...')
    parser = init_model()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_args(args)

    os.makedirs(args.save_model_path, exist_ok=True)

    data = Data(args)
    args.label2id = {label: i for i, label in enumerate(data.all_label_list)}
    args.id2label = {i: label for i, label in enumerate(data.all_label_list)}

    print(args)


    # step1
    external_pretrain = os.path.join("./save_models/external_pretrain.pth")
    manager_p = None
    manager_ext = ExternalPretrainModelManager(
        args,
        data.external_train_dataloader,
        data.external_eval_dataloader,
        data.external_num_labels
    )
    if isinstance(manager_ext.model, nn.DataParallel):
        ext_model = manager_ext.model.module
    else:
        ext_model = manager_ext.model
    if os.path.exists(external_pretrain):
        print(f"\nFound saved Step 1 model. Loading...")
        ext_model.load_model(external_pretrain)
        pretrained_model_from_stage1 = manager_ext.model
        print("Step 1 Skipped.")
    else:
        print('\n[Step 1] External Pre-training begin...')
        manager_ext.train(args)
        ext_model.save_model(external_pretrain)
        pretrained_model_from_stage1 = manager_ext.model
        print('[Step 1] Finished.')

    if args.known_cls_ratio == 0:
        args.disable_pretrain = True  # disable internal pretrain
        final_pretrained_model = pretrained_model_from_stage1
        print("[Stage 1] Internal Pre-training SKIPPED (known_cls_ratio is 0).")
    else:
        args.disable_pretrain = False
        print('\n[Step 1] Internal Pre-training begin...')
        manager_p = InternalPretrainModelManager(args, data)

        if args.step2_ckpt and os.path.exists(args.step2_ckpt):
            print(f"Loading cached Step 1 model from: {args.step2_ckpt}")
            manager_p.model.load_model(args.step2_ckpt)
        else:
            if isinstance(pretrained_model_from_stage1, nn.DataParallel):
                pretrained_dict = pretrained_model_from_stage1.module.backbone.state_dict()
            else:
                pretrained_dict = pretrained_model_from_stage1.backbone.state_dict()

            if isinstance(manager_p.model, nn.DataParallel):
                manager_p.model.module.backbone.load_state_dict(pretrained_dict, strict=True)
            else:
                manager_p.model.backbone.load_state_dict(pretrained_dict, strict=True)

            manager_p.train(args, data)

            if args.step2_ckpt:
                print(f"Saving Step 2 model to cache: {args.step2_ckpt}")
                os.makedirs(os.path.dirname(args.step2_ckpt), exist_ok=True)
                manager_p.model.save_model(args.step2_ckpt)

        final_pretrained_model = manager_p.model
        print('Internal Pre-training finished!')


    # step2
    manager_d = NoiseManager(args, data)
    if args.view_strategy == "SADA":
        print('\nLoading Teacher model backbone...')
        if isinstance(final_pretrained_model, nn.DataParallel):
            pretrained_final = final_pretrained_model.module.backbone.state_dict()
        else:
            pretrained_final = final_pretrained_model.backbone.state_dict()

        if isinstance(manager_d.teacher_model, nn.DataParallel):
            manager_d.teacher_model.module.backbone.load_state_dict(pretrained_final, strict=True)
        else:
            manager_d.teacher_model.backbone.load_state_dict(pretrained_final, strict=True)

        final_model_dict = final_pretrained_model.state_dict()

        if 'classifier.weight' in final_model_dict:
            print("Loading classifier...")
            ckpt_weight = final_model_dict['classifier.weight']
            ckpt_bias = final_model_dict.get('classifier.bias', None)
            model_classifier = manager_d.teacher_model.classifier
            src_label_list = data.known_label_list
            tgt_label_map = args.label2id
            loaded_count = 0
            with torch.no_grad():
                for src_idx, label_name in enumerate(src_label_list):
                    if label_name in tgt_label_map:
                        tgt_idx = tgt_label_map[label_name]
                        if model_classifier.weight.shape[1] == ckpt_weight.shape[1]:
                            model_classifier.weight[tgt_idx] = ckpt_weight[src_idx]
                            if ckpt_bias is not None and model_classifier.bias is not None:
                                model_classifier.bias[tgt_idx] = ckpt_bias[src_idx]
                            loaded_count += 1
                        else:
                            print(f" [Error] Dimension mismatch for label '{label_name}'")
                    else:
                        pass
            print(f"Teacher Classifier Head: {loaded_count}/{len(tgt_label_map)} classes matched and loaded.")
            print(f"(Remaining {len(tgt_label_map) - loaded_count} classes are randomly initialized)")
        else:
            print("[Warning] No classifier found in checkpoint. Using random initialization for Head.")

        if args.step3_ckpt and os.path.exists(args.step3_ckpt):
            print(f"\n[Step 2] Found cached Student Model at: {args.step3_ckpt}")
            manager_d.student_model.load_state_dict(torch.load(args.step3_ckpt))
        else:
            print('\n[Step 2] Distillation begin...')
            manager_d.Mask_BERT_with_ratio(args, data)
            print('\n[Step 2] Distillation finish...')

            if args.step3_ckpt:
                print(f"Saving Step 3 Student Model to: {args.step3_ckpt}")
                os.makedirs(os.path.dirname(args.step3_ckpt), exist_ok=True)
                torch.save(manager_d.student_model.state_dict(), args.step3_ckpt)
    else:
        print(f"Strategy is '{args.view_strategy}', skipping Step 2.")

    s_model = None
    if args.view_strategy == "SADA":
        s_model = manager_d.student_model
    manager_sada = SADAModelManager(args, data, final_pretrained_model, student_model=s_model)


    if args.report_pretrain:
        method = args.method
        args.method = 'pretrain_eval'
        print('Evaluating model performance before training...')
        manager_sada.evaluation(args, data)
        args.method = method

    print('\n[Step 3] SADA Training begin...')
    manager_sada.train(args, data)
    print('[Step 3] Finished.')

    print('\n Saving Final Model...')
    manager_sada.model.save_backbone(args.save_model_path)
    print(f"\n Final model saved to {args.save_model_path}")

    print('\n Final Evaluation begin...')
    manager_sada.evaluation(args, data)

    print("\n Visualizing learned features...")
    manager_sada.visualize_Tsne(data=data,args=args)

    print("\n All Finished!")
