import torch.nn as nn
import torch.optim as optim

import configparser

from info_clusters.encoders import param_reader
from info_clusters.encoders.model_utils import *
from info_clusters.encoders.model_utils import p_r_f, print_result_summary, log_params
from sklearn.metrics import confusion_matrix
import uuid

UNK = 'UNK'

class CNN(nn.Module):


    def __init__(self, labelset_size, embedding_dim, num_feature_maps, kernel_size, pretrained_embeddings, vocab_size, dropout, stride=1, padding=1, dilation=1):
        super(CNN, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_feature_maps = num_feature_maps
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dropout = dropout

        # define the layers
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=num_feature_maps, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(in_features=num_feature_maps, out_features=labelset_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout_layer = nn.Dropout(p=dropout)

        self.init_emb(pretrained_embeddings)


    def forward(self, x):

        #embedding
        embedded_seqs = self.embedding(x)

        # transpose the embedded sequences such that it fits [bs, in_channels, seq_len]
        embedded_seqs_transposed = embedded_seqs.transpose(dim0=1, dim1=2)

        # convolution
        out = torch.nn.ReLU()(self.conv(embedded_seqs_transposed))

        # max pooling
        pooled_output = nn.MaxPool1d(kernel_size=x.data.shape[-1])(out)

        # transpose the pooled output such that it fits [bs, _, num_feature_maps]
        # transposed_output = pooled_out.reshape(pooled_out.size(0), -1)
        transposed_output = pooled_output.transpose(dim0=1, dim1=2)
        output = self.fc1(transposed_output)
        output= output.squeeze(1)
        output = self.softmax(output)

        return output

    def init_emb(self, pretrained_embeddings):
        if pretrained_embeddings is not None:
            self.embedding.weight = torch.nn.Parameter(torch.Tensor(pretrained_embeddings))
        else:
            self.embedding.weight.data.normal_(0, 0.1)



def evaluate_validation_set(model, seqs, golds, lengths, sentences ,criterion, labelset, compute_auc=False):
    y_true = list()
    y_pred = list()
    y_probs = list()
    total_loss = 0
    for batch, targets, lengths, raw_data, _ in seqs2minibatches(seqs, golds, lengths, sentences, batch_size=1):

        batch, targets, lengths = sort_batch(batch, targets, lengths)
        pred = model(batch)
        loss = criterion(pred, targets)
        pred_idx = torch.max(pred, 1)[1]
        y_probs.append(torch.exp(pred))
        y_true += list(targets.int())
        y_pred += list(pred_idx.data.int())
        total_loss += loss

    results = p_r_f(y_true, y_pred, labelset)
    cm = confusion_matrix(y_true, y_pred, labels=[i for i in range(len(labelset))])
    results['cm']= cm
    if compute_auc is True:
        y_true = [int(elm) for elm in y_true]
        aucs, precs, recs, thr = get_auc(y_true, torch.cat(y_probs, dim=0).detach().numpy(), labelset)
        results['aucs'] = aucs
    return total_loss.data.float()/len(seqs), results


def write_predictions(model, seqs, golds, lengths, sentences, tidss, labelset, fname, write_probs=False):
    """

    :param model:
    :param seqs:
    :param golds:
    :param lengths:
    :param sentences:
    :param tidss:
    :param labelset:
    :param fname:
    :param write_probs: if True, write the prediction probabilities
    :return:
    """
    y_true = list()
    y_pred = list()
    raw = []
    tids = []
    preds = []
    for batch, targets, lengths, raw_batch, tids_batch in seqs2minibatches(seqs, golds, lengths, sentences, tids=tidss, batch_size=1):
        pred = model(batch)
        preds.append(torch.exp(pred).detach().numpy())
        pred_idx = torch.max(pred, 1)[1]
        y_true += list(targets.int())
        y_pred += list(pred_idx.data.int())
        raw += raw_batch
        tids += tids_batch

    outlines = [['tid', 'seq', 'gold', 'pred', 'TP']]

    for tid, sent, gold, pred in zip(tids, raw, y_true, y_pred):
        outlines.append(['#'+tid, sent, labelset[gold], labelset[pred], int(gold==pred)])

    if write_probs is True:
        outlines[0].extend(labelset)
        for pred, outline in zip(preds, outlines[1:]):

            outline.extend([elm for elm in pred[0]])

    param_reader.write_csv(fname, outlines)
    return outlines




def main(args):

    # read params from csv and update the arguments
    if args.hyperparam_csv != '':
        csv_params = param_reader.read_hyperparams_from_csv(args.hyperparam_csv, args.rowid)
        vars(args).update(csv_params)


    seed = args.seed
    num_epochs = args.epochs
    batch_size = args.bs
    embedding_dim = args.emb_dim
    num_feature_maps = args.num_feature_maps
    kernel_size = args.ks
    lr = args.lr
    p = args.dropout
    use_pretrained_embeddings = args.embs
    datafile = args.data
    max_vocab = args.max_vocab
    additional_data_file = args.additional_data

    torch.manual_seed(seed)
    np.random.seed(seed)

    random_name = uuid.uuid4().hex
    setup_logging(logfile='{}.log'.format(random_name))
    pred_file = os.path.join(args.pred_dir, '{}.preds'.format(random_name))

    vars(args).update({'pred_file':pred_file})
    log_params(vars(args))

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(args.config)

    feature_extractor = sents2seqs

    if use_pretrained_embeddings is True:
        embeddings_file = config.get('Files', 'emb_file')
        logging.info('Loading pretrained embedding from {}'.format(embeddings_file))
        pretrained_embeddings, word2idx, idx2word = load_embeddings_from_file(embeddings_file, max_vocab=max_vocab)
    else:
        pretrained_embeddings, word2idx, idx2word = None, None, None

    d = load_json(datafile)

    if additional_data_file != '':
        additional_data_file = config.get('Files', 'additional_data_{}'.format(args.additional_data))
        logging.info('Loading additional data from {}'.format(additional_data_file))
        additional_data = load_json(additional_data_file)
    else:
        additional_data = {'seq':[], 'label':[]}
    sentences = [prefix_sequence(sent, 'en', strip_hs=args.strip) for sent in d['train']['seq']] + [prefix_sequence(sent, 'ru', strip_hs=args.strip) for sent in additional_data['seq']]
    labels = d['train']['label'] + additional_data['label']

    if args.upsample is True:
        logging.info('Upsampling the train data')
        sentences, labels = upsample(sentences, labels)


    dev_sentences = [prefix_sequence(sent, 'en', strip_hs=args.strip) for sent in d['dev']['seq']]
    dev_labels = d['dev']['label']
    dev_tids = d['dev']['tid']
    dev_raw_sentences = d['dev']['seq']

    # prepare train set
    seqs, lengths, word2idx = feature_extractor(sentences, word2idx)
    logging.info('Vocabulary has {} entries'.format(len(word2idx)))
    logging.info(word2idx)
    golds, labelset = prepare_labels(labels, None)

    # prepare dev set
    dev_seqs, dev_lengths, _ = feature_extractor(dev_sentences, word2idx)
    dev_golds, _ = prepare_labels(dev_labels, labelset)




    model = CNN(embedding_dim=embedding_dim, num_feature_maps=num_feature_maps, pretrained_embeddings=pretrained_embeddings, kernel_size=kernel_size,
                vocab_size=len(word2idx), labelset_size=len(labelset), dropout=p)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    dev_loss, results = evaluate_validation_set(model=model, seqs=dev_seqs, golds=dev_golds, lengths=dev_lengths,
                                                sentences=dev_sentences, criterion=loss_function, labelset=labelset)

    logging.info('Epoch {}: val f_macro {:.4f}'.format(-1, results['macro'][2]))
    logging.info('Summary val')
    logging.info(print_result_summary(results))

    best_epoch = 0
    best_macro_f = 0
    for epoch in range(num_epochs):
        preds = []
        gold_labels = []
        total_loss = 0
        for seqs_batch, gold_batch, lengths_batch, raw_batch, _ in seqs2minibatches(seqs, golds, lengths, sentences, batch_size=batch_size):
            seqs_batch, gold_batch, lengths_batch = sort_batch(seqs_batch, gold_batch, lengths_batch)
            model.zero_grad()
            out = model(seqs_batch)
            loss = loss_function(out, gold_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss

            pred_idx = torch.max(out, 1)[1]
            gold_labels += list(gold_batch.int())
            preds += list(pred_idx.data.int())

        # predict the train data
        train_results = p_r_f(gold_labels, preds, labelset)
        # predict the val data

        dev_loss, dev_results = evaluate_validation_set(model=model, seqs=dev_seqs, golds=dev_golds, lengths=dev_lengths,
                                                    sentences=dev_sentences, criterion=loss_function, labelset=labelset)
        if dev_results['macro'][2] > best_macro_f:
            best_macro_f = dev_results['macro'][2]
            best_epoch = epoch

        logging.info('Epoch {}: Train loss {:.4f}, train f_macro {:.4f}, val loss {:.4f},  val f_macro {:.4f}, best_epoch {}, best val_f_macro {:.4f}'.format(epoch, total_loss.data.float()/len(seqs), train_results['macro'][2], dev_loss, dev_results['macro'][2],
                                                                                                                                                                                                          best_epoch, best_macro_f))

        logging.info('Summary train')
        logging.info(print_result_summary(train_results))
        logging.info('Summary dev')
        logging.info(print_result_summary(dev_results))

        #logging.info('Summary dev_up')
        #logging.info(print_result_summary(dev_results_up))

    dev_loss, dev_results = evaluate_validation_set(model=model, seqs=dev_seqs, golds=dev_golds,
                                                              lengths=dev_lengths,
                                                              sentences=dev_sentences, criterion=loss_function,
                                                              labelset=labelset, compute_auc=True)
    logging.info(print_auc_summary(dev_results['aucs'], labelset))

    dev_results['best_epoch'] = best_epoch
    dev_results['best_macro_f'] = best_macro_f
    param_reader.write_results_and_hyperparams(args.result_csv, dev_results, vars(args), labelset)
    write_predictions(model, dev_seqs, dev_golds, dev_lengths, dev_raw_sentences, dev_tids, labelset, pred_file, write_probs=False)

    if args.predict_test is True:
        # Prepare test data
        test_sentences = [prefix_sequence(sent, 'en', strip_hs=args.strip) for sent in d['test']['seq']]
        test_labels = d['test']['label']

        test_seqs, test_lengths, _ = feature_extractor(test_sentences, word2idx)
        test_golds, _ = prepare_labels(test_labels, labelset)
        test_tids = d['test']['tid']
        test_raw_sentences = d['test']['seq']
        test_loss, test_results = evaluate_validation_set(model=model, seqs=test_seqs, golds=test_golds,
                                                        lengths=test_lengths,
                                                        sentences=test_sentences, criterion=loss_function,
                                                        labelset=labelset, compute_auc=True)
        logging.info('Summary test')
        logging.info(print_result_summary(test_results))
        param_reader.write_results_and_hyperparams(args.test_result_csv, test_results, vars(args), labelset)
        write_predictions(model, test_seqs, test_golds, test_lengths, test_raw_sentences, test_tids, labelset, pred_file + '.test',
                          write_probs=True)

    if args.predict_all is True:
        # prepare the data to be predicted
        pred_data = load_json(config.get('Files', 'unlabeled'))
        test_sentences = [prefix_sequence(sent, 'en', strip_hs=args.strip) for sent in pred_data['seq']]
        test_seqs, test_lengths, _ = feature_extractor(test_sentences, word2idx)

        test_tids = pred_data['tid']
        test_raw_sentences = pred_data['seq']
        logging.info('Predicting the unlabeled data')
        write_predictions(model, test_seqs, torch.LongTensor(np.array([0 for elm in test_seqs])), test_lengths, test_raw_sentences, test_tids, labelset,
                          pred_file + '.unlabeled',
                          write_probs=True)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(
          description='Tweet classification using CNN')
    parser.add_argument('--data', type=str, default='/home/mareike/PycharmProjects/catPics/data/twitter/mh17/experiments/mh17_60_20_20.json',
                            help="File with main data")
    parser.add_argument('--additional_data', type=str,
                        default='combi',
                        choices = ['', 'combi', 'sbil', 'skaschi', 'samo'],
                        help="Additional train data. if empty string, no additional data is used")
    parser.add_argument('--config', type=str, default='config.cfg',
                        help="Config file")
    parser.add_argument('--exp_dir', type=str, default='out',
                        help="Path to experiment folder")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed")
    parser.add_argument('--bs', type=int, default=256,
                        help="Batch size")
    parser.add_argument('--epochs', type=int, default=500,
                        help="Number of epochs")
    parser.add_argument('--emb_dim', type=int, default=300,
                        help="Embedding dimension")
    parser.add_argument('--num_feature_maps', type=int, default=10,
                        help="Number of CNN feature maps")
    parser.add_argument('--ks', type=int, default=3,
                        help="kernel size")
    parser.add_argument('--upsample', type=bool_flag, default=True,
                        help="if enabled upsample the train data according to size of largest class")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument('--dropout', type=float, default=0.2,
                        help="Keep probability for dropout")
    parser.add_argument('--embs', type=bool, default=True,
                        help="Use pre-trained embeddings")
    parser.add_argument('--max_vocab', type=int, default=-1,
                        help="Maximum number of words read in from the pretrained embeddings. -1 to disable")
    parser.add_argument('--hyperparam_csv', type=str,
                        default='../hyperparams.csv',
                        help="File with hyperparams. If set, values for specified hyperparams are read from the csv")
    parser.add_argument('--result_csv', type=str,
                        default='../results_pc.csv',
                        help="File the results and hyperparams are written to")
    parser.add_argument('--test_result_csv', type=str,
                        default='../results_pc_test.csv',
                        help="File the results and hyperparams are written to")
    parser.add_argument('--pred_dir', type=str,
                        default='predictions',
                        help="Directory storing the prediction files")
    parser.add_argument('--activation', type=str,
                        default='', choices = ['', 'relu'],
                        help="Activation function")
    parser.add_argument('--rowid', type=int,
                        default=2,
                        help="Row from which hyperparams are read")
    parser.add_argument('--predict_test', type=bool_flag,
                        default=True,
                        help="Predict the test set")
    parser.add_argument('--predict_all', type=bool_flag,
                        default=False,
                        help="Predict the set of all tweets")
    parser.add_argument('--strip', type=bool_flag,
                        default=True,
                        help="Strip hashtags from words to attempt reducing oov")
    args = parser.parse_args()
    main(args)