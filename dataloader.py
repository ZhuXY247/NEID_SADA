import math
from utils.tools import *

max_seq_lengths = {'clinc':30, 'talkmoves':85, 'tscc':85, 'mathdial':85}

class Data:
    def __init__(self, args):
        set_seed(args.seed)
        processor = DatasetProcessor()
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        if args.internal_dataset in max_seq_lengths:
            args.internal_max_seq_length = max_seq_lengths[args.internal_dataset]

        print(f"Loading INTERNAL dataset: {args.internal_dataset}")
        self.internal_data_dir = os.path.join(args.internal_data_dir, args.internal_dataset)
        self.all_label_list = processor.get_labels(self.internal_data_dir)
        self.n_known_cls = math.ceil(len(self.all_label_list) * args.known_cls_ratio)
        if self.n_known_cls > 0:
            self.known_label_list = list(
                np.random.choice(np.array(self.all_label_list), self.n_known_cls, replace=False))
        else:
            self.known_label_list = []

        self.num_labels = len(self.all_label_list)

        # if os.path.exists("debug_splicing_view.txt"):
        #     os.remove("debug_splicing_view.txt")

        self.train_labeled_examples, self.train_unlabeled_examples = self.get_examples(processor, args, 'train')
        print('num_INTERNAL_labeled_samples', len(self.train_labeled_examples))
        print('num_INTERNAL_unlabeled_samples', len(self.train_unlabeled_examples))
        self.eval_examples = self.get_examples(processor, args, 'eval')
        self.test_examples = self.get_examples(processor, args, 'test')

        if self.n_known_cls > 0:
            label_map = {label: i for i, label in enumerate(self.known_label_list)}
            counts = np.zeros(self.n_known_cls)
            for example in self.train_labeled_examples:
                if example.label in label_map:
                    counts[label_map[example.label]] += 1
            epsilon = 1e-5
            weights = 1.0 / (counts + epsilon)
            weights = weights / np.mean(weights)
            self.class_weights = torch.FloatTensor(weights).to(args.device)
            print(f"Class weights: {self.class_weights}")
            self.train_labeled_dataloader = self.get_loader(
                self.train_labeled_examples, args, 'train',
                max_seq_length=args.internal_max_seq_length,
                label_list=self.known_label_list,
                batch_size=args.pretrain_batch_size,
                tokenizer=tokenizer
            )

            self.train_distillation_dataloader = self.get_loader(
                self.train_labeled_examples, args, 'train',
                max_seq_length=args.internal_max_seq_length,
                label_list=self.known_label_list,
                batch_size=args.distillation_batch_size,
                tokenizer=tokenizer
            )
        else:
            self.train_labeled_dataloader = None
            self.train_distillation_dataloader = None
            self.class_weights = None

        self.semi_input_ids, self.semi_input_mask, self.semi_segment_ids, self.semi_label_ids, self.semi_special_mask = self.get_semi(
            self.train_labeled_examples, self.train_unlabeled_examples, args, tokenizer)
        self.train_semi_dataset, self.train_semi_dataloader = self.get_semi_loader(
            self.semi_input_ids, self.semi_input_mask, self.semi_segment_ids, self.semi_label_ids, self.semi_special_mask, args)

        self.eval_dataloader = self.get_loader(
            self.eval_examples, args, 'eval',
            max_seq_length=args.internal_max_seq_length,
            label_list=self.known_label_list,
            batch_size=args.eval_batch_size,
            tokenizer=tokenizer
        )
        self.test_dataloader = self.get_loader(
            self.test_examples, args, 'test',
            max_seq_length=args.internal_max_seq_length,
            label_list=self.all_label_list,
            batch_size=args.eval_batch_size,
            tokenizer=tokenizer
        )

        print(f"Loading External Dataset ({args.external_dataset}) for Pre-training...")
        original_strategy = args.input_strategy
        original_with_speaker = args.with_speaker

        args.input_strategy = 'ORIGINAL'
        args.with_speaker = False

        self.external_data_dir = os.path.join(args.external_data_dir, args.external_dataset)
        self.external_label_list = processor.get_labels(self.external_data_dir)
        self.external_num_labels = len(self.external_label_list)

        external_train_examples = processor.get_examples(self.external_data_dir, 'train')
        external_eval_examples = processor.get_examples(self.external_data_dir, 'eval')

        self.external_train_dataloader = self.get_loader(
            external_train_examples, args, 'train',
            max_seq_length=args.external_max_seq_length,
            label_list=self.external_label_list,  # 使用全标签 (遵从指令1)
            batch_size=args.pretrain_batch_size,
            tokenizer=tokenizer
        )

        self.external_eval_dataloader = self.get_loader(
            external_eval_examples, args, 'eval',
            max_seq_length=args.external_max_seq_length,
            label_list=self.external_label_list,  # 使用全标签 (遵从指令1)
            batch_size=args.eval_batch_size,
            tokenizer=tokenizer
        )
        print(
            f"Loaded {len(external_train_examples)} external train examples and {len(external_eval_examples)} external eval examples.")

        args.input_strategy = original_strategy
        args.with_speaker = original_with_speaker

    def get_examples(self, processor, args, mode='train'):
        ori_examples = processor.get_examples(self.internal_data_dir, mode)

        if mode == 'train':
            train_labels = np.array([example.label for example in ori_examples])
            train_labeled_ids = []
            if self.known_label_list == []:
                train_labeled_examples = []
                train_unlabeled_examples = copy.deepcopy(ori_examples)
            else:
                for label in self.known_label_list:
                    num = math.ceil(len(train_labels[train_labels == label]) * args.labeled_ratio)
                    pos = list(np.where(train_labels == label)[0])
                    train_labeled_ids.extend(random.sample(pos, num))

                train_labeled_examples, train_unlabeled_examples = [], []
                for idx, example in enumerate(ori_examples):
                    if idx in train_labeled_ids:
                        train_labeled_examples.append(example)
                    else:
                        train_unlabeled_examples.append(example)

            return train_labeled_examples, train_unlabeled_examples

        elif mode == 'eval':
            eval_examples = []
            for example in ori_examples:
                if example.label in self.known_label_list:
                    eval_examples.append(example)
            return eval_examples

        elif mode == 'test':
            return ori_examples

        else:
            raise NotImplementedError(f"Mode {mode} not found")

    def get_semi(self, labeled_examples, unlabeled_examples, args, tokenizer):

        if self.n_known_cls > 0:
            labeled_features = convert_examples_to_features(args, labeled_examples, self.known_label_list,
                                                            args.internal_max_seq_length, tokenizer)
            labeled_input_ids = torch.tensor([f.input_ids for f in labeled_features], dtype=torch.long)
            labeled_input_mask = torch.tensor([f.input_mask for f in labeled_features], dtype=torch.long)
            labeled_segment_ids = torch.tensor([f.segment_ids for f in labeled_features], dtype=torch.long)
            labeled_label_ids = torch.tensor([f.label_id for f in labeled_features], dtype=torch.long)
            labeled_special_mask = torch.tensor([f.special_tokens_mask for f in labeled_features], dtype=torch.long)
        else:
            labeled_features = None

        unlabeled_features = convert_examples_to_features(args, unlabeled_examples, self.all_label_list,
                                                          args.internal_max_seq_length, tokenizer)
        unlabeled_input_ids = torch.tensor([f.input_ids for f in unlabeled_features], dtype=torch.long)
        unlabeled_input_mask = torch.tensor([f.input_mask for f in unlabeled_features], dtype=torch.long)
        unlabeled_segment_ids = torch.tensor([f.segment_ids for f in unlabeled_features], dtype=torch.long)
        unlabeled_label_ids = torch.tensor([-1 for f in unlabeled_features], dtype=torch.long)
        unlabeled_special_mask = torch.tensor([f.special_tokens_mask for f in unlabeled_features], dtype=torch.long)

        if self.n_known_cls > 0:
            semi_input_ids = torch.cat([labeled_input_ids, unlabeled_input_ids])
            semi_input_mask = torch.cat([labeled_input_mask, unlabeled_input_mask])
            semi_segment_ids = torch.cat([labeled_segment_ids, unlabeled_segment_ids])
            semi_label_ids = torch.cat([labeled_label_ids, unlabeled_label_ids])
            semi_special_mask = torch.cat([labeled_special_mask, unlabeled_special_mask])

        else:
            semi_input_ids = unlabeled_input_ids
            semi_input_mask = unlabeled_input_mask
            semi_segment_ids = unlabeled_segment_ids
            semi_label_ids = unlabeled_label_ids
            semi_special_mask = unlabeled_special_mask

        return semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids, semi_special_mask

    def get_semi_loader(self, semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids, semi_special_mask, args):
        indices = torch.arange(len(semi_input_ids), dtype=torch.long)
        semi_data = TensorDataset(semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids, semi_special_mask, indices)
        semi_sampler = SequentialSampler(semi_data)
        semi_dataloader = DataLoader(semi_data, sampler=semi_sampler, batch_size=args.train_batch_size)

        return semi_data, semi_dataloader

    def get_loader(self, examples, args, mode, max_seq_length, label_list, batch_size, tokenizer):

        # 特征转换
        features = convert_examples_to_features(args, examples, label_list, max_seq_length, tokenizer)

        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        special_tokens_mask = torch.tensor([f.special_tokens_mask for f in features], dtype=torch.long)
        indices = torch.arange(len(features), dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids, special_tokens_mask, indices)

        if mode == 'train':
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        elif mode in ["eval", "test"]:
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=args.eval_batch_size)
        else:
            raise NotImplementedError(f"Mode {mode} not found")

        return dataloader


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, speaker, text_pre, text_a, text_next, text_b=None, label=None, speaker_pre=None, speaker_next=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            speaker: Speaker of the current utterance (text_a).
            text_pre: Content of the previous utterance.
            text_a: Content of the current utterance.
            text_next: Content of the next utterance.
            text_b: Optional.
            label: Label string.
            speaker_pre: Speaker of the previous utterance.
            speaker_next: Speaker of the next utterance.
        """
        self.guid = guid
        self.speaker = speaker
        self.speaker_pre = speaker_pre
        self.speaker_next = speaker_next
        self.text_pre = text_pre
        self.text_a = text_a
        self.text_next = text_next
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, special_tokens_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.special_tokens_mask = special_tokens_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                line = [l.lower() for l in line]
                lines.append(line)
            return lines

class DatasetProcessor(DataProcessor):

    def get_examples(self, data_dir, mode):
        if mode == 'train':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        elif mode == 'eval':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "train")
        elif mode == 'test':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
        else:
            raise NotImplementedError(f"Mode {mode} not found")

    def get_labels(self, data_dir):
        # 1. 过滤列表 (必须与 _create_examples 保持一致)
        ignore_labels = {'none', 'non-english', '0.0', '0'}

        labels = []
        docs = os.listdir(data_dir)

        if "train.tsv" in docs:
            lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
            for i in range(1, len(lines)):
                line = lines[i]
                if len(line) < 2: continue
                if len(line) >= 3:
                    label = line[2]
                else:
                    label = line[1]
                label_clean = str(label).lower().strip()
                if label_clean not in ignore_labels:
                    labels.append(label_clean)
            labels = np.unique(np.array(labels))
        elif "dataset.json" in docs:
            with open(os.path.join(data_dir, "dataset.json"), 'r') as f:
                dataset = json.load(f)
                dataset = dataset[list(dataset.keys())[0]]
            labels = []
            for dom in dataset:
                for ind, data in enumerate(dataset[dom]):
                    label = data[1][0]
                    clean_label = str(label).lower().strip()
                    if clean_label not in ignore_labels:
                        labels.append(clean_label)
            labels = np.unique(np.array(labels))
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i in range(1, len(lines)):
            line = lines[i]
            if len(line) < 2:
                continue

            guid = "%s-%s" % (set_type, i)
            if len(line) >= 3:
                speaker = line[0]
                text_a = line[1]
                label = line[2]
            else:
                speaker = "unk"
                text_a = line[0]
                label = line[1]

            label = str(label).lower().strip()
            if label in ['none', 'non-english', '0.0', '0']:
                continue

            text_pre = None
            speaker_pre = None
            text_next = None
            speaker_next = None

            if i > 1:
                prev_line = lines[i - 1]
                if len(prev_line) >= 3:
                    speaker_pre = prev_line[0]
                    text_pre = prev_line[1]
                elif len(prev_line) >= 2:
                    speaker_pre = "unk"
                    text_pre = prev_line[0]

            if i < len(lines) - 1:
                next_line = lines[i + 1]
                if len(next_line) >= 3:
                    speaker_next = next_line[0]
                    text_next = next_line[1]
                elif len(next_line) >= 2:
                    speaker_next = "unk"
                    text_next = next_line[0]

            examples.append(
                InputExample(
                    guid=guid,
                    speaker=speaker,
                    text_pre=text_pre,
                    text_a=text_a,
                    text_next=text_next,
                    text_b=None,
                    label=label,
                    speaker_pre=speaker_pre,
                    speaker_next=speaker_next
                ))

        return examples


def _truncate_seq_context(tokens_prev, tokens_curr, tokens_next, max_length):
    """
    截断逻辑：
    1. 优先完全保留 tokens_curr (当前句)。
    2. 如果 tokens_curr 本身就超过 max_length，则截断 tokens_curr。
    3. 如果加上前后句超过 max_length，则交替截断前一句和后一句，
       通常截断离当前句最远的部分（即前一句的头部和后一句的尾部）。
    """
    if len(tokens_curr) > max_length:
        tokens_curr[:] = tokens_curr[:max_length]
        tokens_prev[:] = []
        tokens_next[:] = []
        return
    while True:
        total_length = len(tokens_prev) + len(tokens_curr) + len(tokens_next)
        if total_length <= max_length:
            break
        if len(tokens_prev) == 0 and len(tokens_next) == 0:
            break
        if len(tokens_prev) > len(tokens_next):
            tokens_prev.pop(0)
        else:
            tokens_next.pop()

def convert_examples_to_features(args, examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens = []
        segment_ids = []
        special_tokens_mask = []
        if args.input_strategy == "CONTEXT":
            tokens_prev = tokenizer.tokenize(example.text_pre) if hasattr(example, 'text_pre') and example.text_pre else []
            tokens_next = tokenizer.tokenize(example.text_next) if hasattr(example, 'text_next') and example.text_next else []
            # [CLS] ... [SEP] ... [SEP] ... [SEP]
            # 基础结构: CLS, SEP(前), SEP(中), SEP(后), SEP(尾) -> 5个特殊token
            # 如果有 Speaker,构建Speaker:当前句
            if args.with_speaker:
                # speaker_pre = example.speaker_pre if hasattr(example, 'speaker_pre') and example.speaker_pre else "unk"
                # speaker_next = example.speaker_next if hasattr(example, 'speaker_next') and example.speaker_next else "unk"
                # tokens_prev = ["[", speaker_pre, "]"] + tokens_prev
                tokens_a = ["[", example.speaker, "]"] + tokens_a
                # tokens_next = ["[", speaker_next, "]"] + tokens_next

            _truncate_seq_context(tokens_prev, tokens_a, tokens_next, max_seq_length - 4)
            # 前一句
            tokens.append("[CLS]")
            segment_ids.append(0)
            special_tokens_mask.extend([1])
            if tokens_prev:
                tokens.extend(tokens_prev)
            tokens.append("[SEP]")
            segment_ids.extend([0] * (len(tokens_prev) +1))
            special_tokens_mask.extend([0] * len(tokens_prev) + [1])

            # 当前句
            tokens.extend(tokens_a)
            tokens.append("[SEP]")
            segment_ids.extend([1] * (len(tokens_a) + 1))
            special_tokens_mask.extend([0] * len(tokens_a) + [1])
            # 后一句
            if tokens_next:
                tokens.extend(tokens_next)
            tokens.append("[SEP]")
            segment_ids.extend([0] * (len(tokens_next) + 1))
            special_tokens_mask.extend([0] * len(tokens_next) + [1])

        elif args.input_strategy == "ORIGINAL":
            if args.with_speaker:
                tokens_a = ["[" ,example.speaker, "]"] + tokens_a
            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            special_tokens_mask.extend([1] + [0] * len(tokens_a) + [1])

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)
                special_tokens_mask.extend( [0] * len(tokens_b) + [1])

        # if ex_index < 5:
        #     with open("debug_splicing_view.txt", "a", encoding="utf-8") as f:
        #         f.write(f"*** GUID: {example.guid} ***\n")
        #         f.write(f"Strategy: {args.input_strategy}, With Speaker: {args.with_speaker}\n")
        #         f.write(f"Tokens (Unpadded): {' '.join(tokens)}\n")
        #         f.write(f"Segment IDs (Padded): {segment_ids}\n")
        #         f.write(f"Special Tokens Mask (Padded): {special_tokens_mask}\n")
        #         f.write("=" * 50 + "\n")

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * padding_length
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        special_tokens_mask += [1] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        label_id = label_map[example.label]
        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          special_tokens_mask=special_tokens_mask))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  # For dialogue context
        else:
            tokens_b.pop()
