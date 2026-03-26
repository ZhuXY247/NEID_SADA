from argparse import ArgumentParser

def init_model():
    parser = ArgumentParser()
    
    parser.add_argument("--internal_data_dir", default='data', type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")

    parser.add_argument("--internal_dataset", default=None, type=str, required=True,
                        help="Name of dataset.")

    parser.add_argument("--internal_max_seq_length", default=30, type=int,
                        help="Maximum sequence length for the INTERNAL dataset.")

    parser.add_argument("--step2_ckpt", type=str, default=None,
                        help="Path to cache/load Step 2 internal pretrained model.")

    parser.add_argument("--step3_ckpt", type=str, default=None,
                        help="Path to cache/load Step 3 student model.")
    
    parser.add_argument("--save_results_path", type=str, default='outputs',
                        help="The path to save results.")

    parser.add_argument("--external_data_dir", default='data_external', type=str,
                        help="The root input data dir for the EXTERNAL pre-training task.")

    parser.add_argument("--external_dataset", default='clinc', type=str,
                        help="Name of EXTERNAL dataset for pre-training (e.g., clinc).")

    parser.add_argument("--external_max_seq_length", default=30, type=int,
                        help="Maximum sequence length for the EXTERNAL dataset.")

    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="The path or name for the pre-trained bert model.")

    parser.add_argument("--tokenizer", default="bert-base-uncased", type=str,
                        help="The path or name for the tokenizer")
    
    parser.add_argument("--feat_dim", default=768, type=int,
                        help="Bert feature dimension.")
    
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Warmup proportion for optimizer.")

    parser.add_argument("--save_model_path", default='./save_models/final/', type=str,
                        help="Path to save model checkpoints. Set to None if not save.")
    
    parser.add_argument("--known_cls_ratio", default=0.75, type=float, required=True,
                        help="The ratio of known classes.")

    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed.")

    parser.add_argument("--method", type=str, default='SADA',
                        help="The name of method.")

    parser.add_argument("--labeled_ratio", default=0.1, type=float,
                        help="The ratio of labeled samples.")
    
    parser.add_argument("--rtr_prob", default=0.25, type=float,
                        help="Probability for random token replacement")

    parser.add_argument("--pretrain_batch_size", default=64, type=int,
                        help="Batch size for pre-training")

    parser.add_argument("--distillation_batch_size", default=128, type=int,
                        help="Batch size for distillation")

    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="Batch size for training.")
    
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size for evaluation.")

    parser.add_argument("--wait_patient", default=20, type=int,
                        help="Patient steps for Early Stop in pretraining.") 

    parser.add_argument("--num_pretrain_epochs", default=100, type=int,
                        help="The pre-training epochs.")

    parser.add_argument("--num_distillate_epochs", default=50, type=int,
                        help="The distillation epochs.")

    parser.add_argument("--num_train_epochs", default=34, type=int,
                        help="The training epochs.")

    parser.add_argument("--lr_pre", default=5e-5, type=float,
                        help="The learning rate for pre-training.")
    
    parser.add_argument("--lr", default=1e-5, type=float,
                        help="The learning rate for training.")
        
    parser.add_argument("--temp", default=0.07, type=float,
                        help="Temperature for contrastive loss")

    parser.add_argument("--view_strategy", default="SADA", type=str,
                        help="Choose from rtr|shuffle|none")

    parser.add_argument("--input_strategy", default="CONTEXT", type=str,
                        help="input_strategy")

    parser.add_argument("--with_speaker", default=1, type=int,
                        help="with_speaker (1 for True, 0 for False)")

    parser.add_argument("--update_per_epoch", default=5, type=int,
                        help="Update pseudo labels after certain amount of epochs")

    parser.add_argument("--report_pretrain", action="store_true",
                        help="Enable reporting performance right after pretrain")
    
    parser.add_argument("--disable_pretrain", action="store_true",
                                    help="Skip Step 1 (Pre-training)")

    parser.add_argument("--topk", default=50, type=int,
                        help="Select topk nearest neighbors")

    parser.add_argument("--grad_clip", default=1, type=float,
                        help="Value for gradient clipping.")

    parser.add_argument("--ratio", default=0.75, type=float,
                        help="Ratio of tokens to keep (IG).")

    parser.add_argument("--alpha", default=1.0, type=float,
                        help="Weight for distillation loss.")

    parser.add_argument("--with_pos", action="store_true",
                        help="Enable POS tagging filter for IG mask.")

    parser.add_argument("--with_abs", action="store_true",
                        help="Use absolute value for IG scores.")

    parser.add_argument("--is_continuous", action="store_true", default=True,
                        help="Force continuous masking (span masking).")

    parser.add_argument("--with_mask", action="store_true",
                        help="Enable masking for visualization/inference.")

    parser.add_argument("--mask_threshold", default=0.3, type=float,
                        help="Threshold for Student model to determine signal vs noise (default: 0.5).")
    return parser
