"""
Use Pytorch neural network to predict reflectance from cell classifications
"""


from numpy import ndarray
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
import torch
from torch import nn

from tqdm import tqdm
import copy
import seaborn as sns
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torchviz import make_dot
from sklearn.manifold import TSNE
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, NoiseTunnel

import captum
captum.__version__
from reflectance_utils import *
from balanced_loss import Loss
import pickle

ACC_COL = "accuracy"
ROC_AUC_COL = "ROC-AUC"
LOSS_COL = "loss"
DSET_COL = "dataset"
EPOCH_COL = "epoch"
MEASURE_COL = "measure"
VAL_COL = "value"
REP_COL = "rep"
WH_COL = "tile_wh"
TRAIN_NAME = "training"
VAL_NAME = "validation"
TEST_NAME = "test"
ALL_NAME = "all"
WITH_SITE_COL = "with_site"

# Output
TEST_CONFUSION = "confusion_matrix_test"
ALL_CONFUSION = "confusion_matrix_overall"
TRAINING = "training_stats"
CONTRIBUTION = "contributions"
SUMMARY = "summary"

CELL_COL_NAME = "cell_type"
INFLUENCE_COL_NAME = "influence"
MEAN_INFLUENECE_COL = f"mean_{INFLUENCE_COL_NAME}"


DATA_DIR = "data"
FIG_DIR = "figures"


def viz_reduced_data(x, str_labels, sites=None):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(str_labels)

    viz_cls = [label_encoder.classes_[x] for x in labels]

    if sites is None:
        dsets = {"all":[x, str_labels]}

    else:
        unique_sites = np.unique(sites)
        dsets = {}
        for s in unique_sites:
            site_idx = [idx for idx in range(len(sites)) if sites[idx] == s]
            site_x = x[site_idx, :]
            site_labels = [viz_cls[i] for i in site_idx]
            dsets[s] = [site_x, site_labels]

    viz_df_list = []
    for s, (site_x, site_labels) in dsets.items():
        print(s)
        if site_x.shape[1] > 2:
            tsne_perplex = np.min([30, site_x.shape[1]-1])
            # distance_metric = 'cosine'
            reducer = TSNE(
                n_components=2,
                perplexity=tsne_perplex,
                # metric=distance_metric,
                init='random',
                learning_rate='auto',
                n_jobs=-1
            )

            # reducer = PCA(n_components=2)

            reduced_coords = reducer.fit_transform(site_x)
            viz_df = pd.DataFrame({
                "x":reduced_coords[:, 0],
                "y":reduced_coords[:, 1],
                "cls":site_labels,
                "site":s
            })
        else:
            viz_df = pd.DataFrame({
                "x":site_x[:, 0],
                "y":site_x[:, 1],
                "cls":site_labels,
                "site":s
            })
        viz_df_list.append(viz_df)

    viz_df = pd.concat(viz_df_list)

    if sites is None:
        sns.scatterplot(viz_df, x="x", y="y", hue="cls", alpha=0.5)
    else:

        g = sns.FacetGrid(viz_df, col="site", hue="cls")
        g.map(sns.scatterplot, "x", "y", alpha=0.5)

    plt.legend()
    # plt.show()


def viz_dataloader(dl, str_labels=None, with_outputs=True):
    n_batches = np.ceil(len(dl.dataset)/dl.batch_size).astype(int)

    n_cls = len(np.unique(dl.dataset.tensors[1]))
    all_cls_counts = np.zeros((n_batches, n_cls+1))
    for i, (batch_x, batch_y) in enumerate(dl):
        all_cls_counts[i, 0] = i
        print(batch_y.unique())
        batch_counter = Counter(batch_y.tolist())
        for cls_idx, cls_counts in batch_counter.items():
            all_cls_counts[i, int(cls_idx)+1] = cls_counts

    actual_batch_sizes = np.sum(all_cls_counts[:, 1:], axis=1)
    if str_labels is not None:
        colnames = ["batch", *str_labels]
    else:
        colnames = ["batch", *[str(x) for x in range(n_cls)]]

    batch_count_df = pd.DataFrame(all_cls_counts, columns=colnames)

    if with_outputs:
        batch_count_plot_df = batch_count_df.melt(id_vars="batch", var_name="cls", value_name="count")
        sns.barplot(batch_count_plot_df, x="batch", y="count", hue="cls")

    return batch_count_df


def sweep_layers_and_neurons(train_data, val_data, n_layers_range, hidden_units_range, n_epoch=1500):

    n_hidden_units_grid, n_layers_grid = np.meshgrid(hidden_units_range, n_layers_range)

    n_hidden_units_array = n_hidden_units_grid.reshape(-1)
    n_layers_array = n_layers_grid.reshape(-1)

    n_tests = len(n_layers_array)
    res_array = np.zeros(n_tests)
    model_list = [None] * n_tests
    for i in tqdm(range(n_tests)):
        n_hidden_units = n_hidden_units_array[i]
        n_layers = n_layers_array[i]
        print(n_hidden_units, n_layers)

        model = TorchClassifier(n_features=N_FEATURES, n_hidden_units=n_hidden_units,
                            n_layers=n_layers, n_cls=N_CLASSES,
                            activation_fxn=activation,
                            optimizer_cls=optimizer_cls,
                            optimizer_init_kwargs=optimizer_init_kwargs,
                            funnel_s=FUNNEL_D,
                            expand_s=EXPAND_D)

        # self = model
        epoch_num_list, train_acc_list, train_loss_list, val_acc_list, val_loss_list = model.train_model(train_dset=train_data,
                                                                                                        val_dset=val_data,
                                                                                                        n_epoch=n_epoch)

        model.load_model("overall")
        train_acc, train_loss, train_roc_auc = model.calc_score(test_data)
        res_array[i] = train_acc
        model_list[i] = model



    best_model_idx = np.argmax(res_array)
    best_model = model_list[best_model_idx]
    best_hn = n_hidden_units_array[best_model_idx]
    best_layers = n_layers_array[best_model_idx]

    return best_model, best_hn, best_layers, best_model_idx, n_hidden_units_array, n_layers_array


def reduce_cls(x, keep_cls_str, other_name):
    """
    Replace mossy and spotty with `other_name`
    """
    new_x = [y if y == keep_cls_str else other_name for y in x]

    return new_x


def get_cb_loss_fxn(y, samples_per_cls=None, beta=0.999, gamma=2.0, focal_loss=True, cb_loss=True):
    """

    beta : float
        hyper parameter for class balanced loss. 0-1

    gamma : float
        hyper parameter for focal loss

    focal_loss : bool
        Whether or not to use focal loss

    cb_loss: bool
        whether or not to use class balanced loss versions
    """
    if samples_per_cls is None:
        _, samples_per_cls = calc_cls_weights_for_unbalanced(y)

    if focal_loss:
        loss_type = "focal_loss"
    else:
        n_cls = len(samples_per_cls)
        if n_cls <= 2:
            loss_type="cross_entropy"
        else:
            loss_type="binary_cross_entropy"

    loss_fxn = Loss(
        loss_type=loss_type,
        beta=beta, # class-balanced loss beta
        fl_gamma=gamma, # focal loss gamma
        samples_per_class=samples_per_cls,
        class_balanced=cb_loss
    )

    return loss_fxn


class NoScaler(MinMaxScaler):
    def __init__(self):
        super().__init__()

    def fit(self, X):
        return X

    def fit_transform(self, X):
        return X

    def transform(self, X):
        return X

    def check_is_fitted(self, *args, **kwargs):
        return None


class LinearActivation(torch.nn.Module):
    # a linear activation function based on y=x
    def forward(self, output):return output


class TorchClassifier(nn.Module):
    def __init__(self, n_features, n_hidden_units, n_layers=4, n_cls=2, activation_fxn=LinearActivation(),
                 optimizer_cls=torch.optim.SGD, optimizer_init_kwargs={"lr":0.1}, funnel_s=0.5, expand_s=2,
                 cls_weights=None, device=None, label_smoothing=0.1, loss_fxn=None, dropout_p=0.2):
        """
        funnel_s : float, int
            Determines shape of funnel layers that go from `n_hidden_units` to `n_cls`.
            Values less than 1 will have an exponential decrease (if funnel_s = 0.5, it would go 25 -> 12 -> 6 -> 3),
            Values greater than 1 will decrease linearly (if funnel_s=2, 10 -> 8 -> 6 -> 4)


        cls_weights : tensor
            weights for each class to use when calculating loss.
            For unbalanced data, giving rare classes more weight may help
            See https://medium.com/@zergtant/use-weighted-loss-function-to-solve-imbalanced-data-classification-problems-749237f38b75

        """
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.n_cls = n_cls
        out_features = self.n_cls
        self.activation_fxn = activation_fxn
        self.expand_layers = None
        self.funnel_layers = None
        self.dropout_p = dropout_p
        self.build_network(n_features=n_features, n_cls=out_features, n_hidden_units=n_hidden_units, n_layers=n_layers, funnel_s=funnel_s, expand_s=expand_s, dropout_p=dropout_p)

        if loss_fxn is None:
            self.loss_fxn = self.get_loss_fxn(weight=cls_weights, label_smoothing=label_smoothing)
        else:
            self.loss_fxn = loss_fxn

        self.scheduler = None
        self.optimizer = self.get_optimizer(optimizer_cls=optimizer_cls,
                                            optimizer_init_kwargs=optimizer_init_kwargs)

        self.best_train_acc = 0
        self.best_train_epoch = -1
        self.best_train_params = None # Parameters that performed best on training data

        self.best_val_acc = 0
        self.best_val_epoch = -1
        self.best_val_params = None # Parameters that performed best on validation data

        self.best_params = None # Parameters that have maximize accuracy of both training and validation datasets
        self.best_epoch = -1
        self.best_train_val_pair_acc = np.array([0, 0])

        self.last_params = None

        self.training_acc = None
        self.val_acc = None
        self.training_loss = None
        self.val_loss = None
        self.training_roc_auc = None
        self.val_roc_auc = None

    def build_network(self, n_features, n_cls, n_hidden_units, n_layers, funnel_s, expand_s, dropout_p):

        # Create expanding layers
        if expand_s > 0:
            if expand_s < 1:
                n_expand_layers = -np.floor(np.log(n_features/n_hidden_units)/np.log(1/expand_s)).astype(int) - 1
                expand_layer_widths = [np.round(n_hidden_units*(expand_s**x)).astype(int) for x in range(1, n_expand_layers+1)]

            elif expand_s >= 1:
                n_expand_layers = np.floor((n_hidden_units-n_features)/expand_s).astype(int)
                expand_layer_widths = [n_hidden_units-np.round(expand_s*x).astype(int) for x in range(1, n_expand_layers+1)][::-1]

            if n_expand_layers <= 0:
                self.expand_layers = None
                last_expand_layer_w = n_hidden_units

            else:
                expand_layer_widths = np.unique([n_features, *expand_layer_widths, n_hidden_units])
                self.expand_layers = nn.ModuleList([nn.Linear(in_features=expand_layer_widths[i], out_features=expand_layer_widths[i+1]) for i in range(len(expand_layer_widths)-1)])
                for expand_layer in self.expand_layers:
                    nn.init.kaiming_normal_(expand_layer.weight.data)

                last_expand_layer_w = expand_layer_widths[-1]
                self.expand_norm_layers = nn.ModuleList([nn.BatchNorm1d(expand_layer_widths[i+1]) for i in range(n_expand_layers)])
                self.expand_drop_out_layers = None
                if dropout_p > 0:
                    self.expand_drop_out_layers = nn.ModuleList([nn.Dropout(dropout_p) for i in range(n_expand_layers)])

        else:
            n_expand_layers = 0
            last_expand_layer_w = n_hidden_units

        if self.expand_layers is None:
            self.layer_1 = nn.Linear(in_features=n_features, out_features=n_hidden_units) # takes in n features (X), produces 5 features
        else:
            self.layer_1 = self.expand_layers[0]

        self.first_norm_layer = nn.BatchNorm1d(self.layer_1.out_features)
        self.first_dropout_layer = None

        if dropout_p > 0:
            dropout_range = np.linspace(dropout_p, 0, n_layers+1)
            self.first_dropout_layer = nn.Dropout(dropout_range[0])

        # Create inner layer
        self.inner_layers = nn.ModuleList([nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units) for i in range(n_layers)])
        for inner_layer in self.inner_layers:
            nn.init.kaiming_normal_(inner_layer.weight.data)
        self.inner_norm_layers = nn.ModuleList([nn.BatchNorm1d(n_hidden_units) for i in range(n_layers)])

        self.inner_drop_out_layers = None
        if dropout_p > 0:
            self.inner_drop_out_layers = nn.ModuleList([nn.Dropout(dropout_range[i+1]) for i in range(n_layers)])

        # Create out funnel layers
        if funnel_s > 0:
            if funnel_s < 1:
                n_funnel_layers = np.floor(np.log(n_cls/n_hidden_units)/np.log(funnel_s)).astype(int) - 1
                funnel_layer_widths = [np.round(n_hidden_units*(funnel_s**x)).astype(int) for x in range(1, n_funnel_layers+1)]

            elif funnel_s >= 1:
                n_funnel_layers = np.floor((n_hidden_units-n_cls)/funnel_s).astype(int)
                funnel_layer_widths = [n_hidden_units-np.round(funnel_s*x).astype(int) for x in range(1, n_funnel_layers+1)]

            if n_funnel_layers <= 0:
                last_funnel_layer_w = n_hidden_units

            else:
                funnel_layer_widths = [n_hidden_units, *funnel_layer_widths]
                self.funnel_layers = nn.ModuleList([nn.Linear(in_features=funnel_layer_widths[i], out_features=funnel_layer_widths[i+1]) for i in range(n_funnel_layers)])
                for funnel_layer in self.funnel_layers:
                    nn.init.kaiming_normal_(funnel_layer.weight.data)

                self.funnel_norm_layers = nn.ModuleList([nn.BatchNorm1d(funnel_layer_widths[i+1]) for i in range(n_funnel_layers)])
                self.funnel_drop_out_layers = None

                last_funnel_layer_w = funnel_layer_widths[-1]
        else:
            last_funnel_layer_w = n_hidden_units

        self.last_layer = nn.Linear(in_features=last_funnel_layer_w, out_features=n_cls)
        nn.init.kaiming_normal_(self.last_layer.weight.data)

    def forward(self, x, return_probs=True):

        z = self.activation_fxn(self.layer_1(x))
        z = self.first_norm_layer(z)
        if self.first_dropout_layer is not None:
            z = self.first_dropout_layer(z)

        # Expanding layers
        if self.expand_layers:
            for i in range(1, len(self.expand_layers)):
                layer = self.expand_layers[i]
                norm_layer = self.expand_norm_layers[i]

                z = self.activation_fxn(layer(z))
                z = norm_layer(z)

        # Inner layers
        for i, layer in enumerate(self.inner_layers):
            norm_layer = self.inner_norm_layers[i]

            z = self.activation_fxn(layer(z))
            z = norm_layer(z)
            if self.inner_drop_out_layers is not None:
                dropout_layer = self.inner_drop_out_layers[i]
                z = dropout_layer(z)

        # Funnel layers
        if self.funnel_layers is not None:
            for i, layer in enumerate(self.funnel_layers):
                norm_layer = self.funnel_norm_layers[i]

                z = self.activation_fxn(layer(z))
                z = norm_layer(z)

                if self.funnel_drop_out_layers is not None:
                    dropout_layer = self.funnel_drop_out_layers[i]
                    z = dropout_layer(z)

        # Output layer
        z = self.last_layer(z)
        if return_probs:
            z = self.logit2prob(z)

        return z

    def logit2prob(self, x):

        if self.last_layer.out_features == 1:
            probs = torch.sigmoid(x.squeeze())
        else:
            probs = torch.softmax(x.squeeze(), dim=1)
            if not all(probs.sum(1)==1):
                probs = probs/probs.sum(axis=1)[[..., None]]
        return probs

    def label_binary(self, x):
        # Need logits for loss function, so don't return probs
        logits = self(x.to(self.device), return_probs=False).squeeze()
        probs = self.logit2prob(logits)
        pred = torch.round(probs)

        return logits, probs, pred

    def label_mc(self, x):
        # Need logits for loss function, so don't return probs
        logits = self(x.to(self.device), return_probs=False).squeeze()
        probs = self.logit2prob(logits)
        pred = torch.argmax(probs, dim=1)

        return logits, probs, pred

    def predict(self, x):
        if self.last_layer.out_features == 1:
            logits, probs, pred = self.label_binary(x)
        else:
            logits, probs, pred = self.label_mc(x)

        return logits, probs, pred

    def get_optimizer(self, optimizer_cls, optimizer_init_kwargs):
        optimizer = optimizer_cls(self.parameters(), **optimizer_init_kwargs)
        return optimizer

    def get_loss_fxn(self, label_smoothing=0, weight=None):
        if self.n_cls == 2:
            pos_w = None if weight is None else weight[1]
            loss_fxn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        else:
            loss_fxn = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

        return loss_fxn

    def load_model(self, dset="overall"):
        """
        Load the best models
        """
        if dset == "overall":
            best_weights = self.best_params
        elif dset == "train":
            best_weights = self.best_train_params
        elif dset == "test":
            best_weights = self.best_val_params
        else:
            print("loading last model")
            best_weights = self.last_params

        self.load_state_dict(best_weights)

    def calc_score(self, dset):
        running_loss = 0.0
        running_total = 0
        running_corrects = 0
        running_roc_auc = 0

        with torch.inference_mode():
            self.eval() # set model in evaluation mode
            for i, (batch_x, batch_y) in enumerate(dset):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                if batch_x.shape[0] == 1:
                    continue
                with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16):
                    logits, probs, pred = self.predict(batch_x)
                    batch_loss = self.calc_loss(logits, batch_y)

                    probs_np = probs.type(torch.double).detach().numpy()
                    probs_np = probs_np/probs_np.sum(axis=1)[..., np.newaxis]
                    roc_auc = roc_auc_score(batch_y.detach().numpy().astype(int), probs_np, multi_class="ovr")

                    running_roc_auc += roc_auc * batch_x.size(0)
                    running_loss += batch_loss.item() * batch_x.size(0)
                    running_total += batch_x.size(0)
                    running_corrects += (pred == batch_y).sum().item()

        avg_loss = running_loss/len(dset.dataset)
        avg_acc = running_corrects / running_total
        avg_roc_auc = running_roc_auc / len(dset.dataset)

        return avg_acc, avg_loss, avg_roc_auc

    def calc_loss(self, logits, true_labels):
        fxn_name = self.loss_fxn.__class__.__name__
        if fxn_name == "BCEWithLogitsLoss":
            loss = self.loss_fxn(logits, true_labels)
        else:
            loss = self.loss_fxn(logits, true_labels.long())

        return loss

    def train_epoch(self, train_dset, val_dset, epoch_id=0, minimize_val_loss=False):
        running_loss = 0.0
        running_corrects = 0
        running_total = 0
        running_roc_auc = 0

        try:
            self.train() # set model in training mode
            for i, (x_batch, y_batch) in enumerate(tqdm(train_dset, unit=" batch", desc=f"Epoch {epoch_id}", leave=None)):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                if x_batch.shape[0] == 1:
                    continue

                with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16):
                    batch_logits, batch_probs, batch_pred = self.predict(x=x_batch)

                    if minimize_val_loss:
                        val_x, val_y = val_dset.dataset.tensors
                        with torch.no_grad():
                            self.eval()
                            val_logits, val_probs, val_pred = self.predict(val_x)

                        self.train()
                        loss = self.calc_loss(val_logits, val_y)
                    else:
                        loss = self.loss_fxn(batch_logits, y_batch.long())
                        loss = self.calc_loss(logits=batch_logits, true_labels=y_batch)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()

                probs_np = batch_probs.type(torch.double).detach().numpy()
                probs_np = probs_np/probs_np.sum(axis=1)[..., np.newaxis]
                roc_auc = roc_auc_score(y_batch.detach().numpy().astype(int), probs_np, multi_class="ovr")

                running_roc_auc += roc_auc*x_batch.size(0)
                running_loss += loss.item() * x_batch.size(0)
                running_total += x_batch.size(0)
                running_corrects += (batch_pred == y_batch).sum().item()

        except Exception as e:
            print(f"Training failed for the following reason: {e}")
            pass

        # Testing
        with torch.inference_mode():
            self.eval() # set model in evaluation mode
            train_loss = running_loss / len(train_dset.dataset)
            train_roc_auc = running_roc_auc / len(train_dset.dataset)
            # train_accuracy = 100 * running_corrects / running_total
            train_accuracy = running_corrects / running_total

            if val_dset is None:
                val_loss = np.inf
                val_acc = 0
                val_roc_auc = 0
            else:
                val_acc, val_loss, val_roc_auc = self.calc_score(val_dset)

        return train_accuracy, train_loss, train_roc_auc, val_acc, val_loss, val_roc_auc

    def train_model(self, train_dset, val_dset, n_epoch=100, record_freq=10, minimize_val_loss=False, use_scheduler=True, overfit_epoch_thresh=25, loss_diff_thresh=0.1, min_n_epoch=50):
        epoch_num_list = [None]
        train_acc_list = [None]
        train_loss_list = [None]
        train_roc_auc_list = [None]

        val_acc_list = [None]
        val_loss_list = [None]
        val_roc_auc_list = [None]

        if use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.optimizer.param_groups[0]["lr"]*10, steps_per_epoch=len(train_dset), epochs=n_epoch)

        if record_freq > 0:
            n_record_steps = n_epoch//record_freq
            if n_epoch % record_freq != 0:
                n_record_steps += 1

            epoch_num_list *= n_record_steps

            train_acc_list *= n_record_steps
            train_loss_list *= n_record_steps
            train_roc_auc_list *= n_record_steps

            val_acc_list *= n_record_steps
            val_loss_list *= n_record_steps
            val_roc_auc_list *= n_record_steps

        all_params = [None] * n_epoch
        all_train_acc = np.zeros(n_epoch)
        all_train_roc_auc = np.zeros(n_epoch)

        all_val_acc = np.zeros(n_epoch)
        all_train_loss = np.zeros(n_epoch)
        all_val_loss = np.zeros(n_epoch)
        all_val_roc_auc = np.zeros(n_epoch)

        n_overfits = 0
        min_val_loss = np.inf
        min_train_loss = np.inf
        from sklearn.linear_model import LinearRegression
        reg_range = np.arange(0, overfit_epoch_thresh).reshape(-1, 1)
        for i in tqdm(range(n_epoch), unit=" epoch", desc=f"Training", leave=None):

            train_acc, train_loss, train_roc_auc, val_acc, val_loss, val_roc_auc = self.train_epoch(train_dset,
                                                                        val_dset,
                                                                        epoch_id=f"{i} (train roc= {all_train_roc_auc[i-1]:.3f}, val roc = {all_val_roc_auc[i-1]:.3f})",
                                                                        minimize_val_loss=minimize_val_loss)

            if val_loss < min_val_loss:
                min_val_loss = val_loss

            if train_loss < min_train_loss:
                min_train_loss = train_loss
            loss_diff = val_loss - train_loss # If too high, then
            if i >= overfit_epoch_thresh + min_n_epoch and loss_diff >= loss_diff_thresh:

                val_fit = LinearRegression().fit(reg_range, all_val_loss[i-overfit_epoch_thresh:i].reshape(-1, 1))
                train_fit = LinearRegression().fit(reg_range, all_train_loss[i-overfit_epoch_thresh:i].reshape(-1, 1))

                if val_fit.coef_[0] > 0 and train_fit.coef_[0] < 0:
                    n_overfits += overfit_epoch_thresh
                    if n_overfits >= 2:
                        print(f"Warning, model may be overfitting. Validation loss is increasing while training loss is decreasing. Has happend {n_overfits} times in a row. Difference in loss is {loss_diff}, which exceeds loss_diff_thresh of {loss_diff_thresh}")
                else:
                    n_overfits = 0

            # Keep most accurate models
            all_params[i] = copy.deepcopy(self.state_dict())
            all_train_acc[i] = train_acc
            all_train_roc_auc[i] = train_roc_auc

            all_val_acc[i] = val_acc
            all_train_loss[i] = train_loss
            all_val_loss[i] = val_loss
            all_val_roc_auc[i] = val_roc_auc

            if train_acc > self.best_train_acc:
                self.best_train_epoch = i
                # Parameters that peformed best on the training data
                self.best_train_acc = train_acc
                self.best_train_params = copy.deepcopy(self.state_dict())

            if val_acc > self.best_val_acc:
                # Parameters that peformed best on the test data
                self.best_val_epoch = i
                self.best_val_acc = val_acc
                self.best_val_params = copy.deepcopy(self.state_dict())

            if record_freq > 0 and (i % record_freq == 0 or i == (n_epoch-1)):
                if i % record_freq == 0:
                    idx = i // record_freq
                else:
                    idx = -1
                epoch_num_list[idx] = i
                train_acc_list[idx] = train_acc
                train_loss_list[idx] = train_loss
                train_roc_auc_list[idx] = train_roc_auc

                val_acc_list[idx] = val_acc
                val_loss_list[idx] = val_loss
                val_roc_auc_list[idx] = val_roc_auc

            if n_overfits >= 5:
                print(f"Stopping, as model likely overfit. Validation loss is increasing while training loss is decreasing. Happend {n_overfits} times in a row")
                break

        self.last_params = copy.deepcopy(self.state_dict())

        self.training_acc = all_train_acc
        self.val_acc = all_val_acc

        self.training_loss = all_train_loss
        self.val_loss = all_val_loss

        self.training_roc_auc = all_train_roc_auc
        self.val_roc_auc = all_val_roc_auc

        train_acc_col = "train_acc"
        val_acc_col = "vall_acc"
        train_loss_col = "train_loss"
        val_loss_col = "val_loss"

        train_roc_auc_col = "train_roc_auc"
        val_roc_auc_col = "val_roc_auc"

        acc_df = pd.DataFrame(
            {train_acc_col:self.training_acc,
            val_acc_col:self.val_acc,
            train_loss_col:self.training_loss,
            val_loss_col:self.val_loss,
            train_roc_auc_col: self.training_roc_auc,
            val_roc_auc_col: self.val_roc_auc,
            })

        best_overall_idx = acc_df.sort_values([train_roc_auc_col, val_roc_auc_col], ascending=[False, False]).index[0]

        self.best_epoch = best_overall_idx
        self.best_train_val_pair_acc = np.array([all_train_acc[best_overall_idx], all_val_acc[best_overall_idx]])
        self.best_params = all_params[best_overall_idx]

        self.eval()

        return epoch_num_list, train_acc_list, train_loss_list, train_roc_auc_list, val_acc_list, val_loss_list, val_roc_auc_list


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Predict relfectance niche from quadrat counts')
    parser.add_argument('-qc_tile_dir', type=str, help='folder containing quadrat counts')
    parser.add_argument('-fig_dst_dir', type=str, help='where to save the figures')
    parser.add_argument('-data_dst_dir', type=str, help='where to save the raw data')
    parser.add_argument('-rep_id', type=str, help='where to save the raw data')
    parser.add_argument('-with_site', type=str, help='where to save the raw data')
    parser.add_argument('-max_epoch', type=int, help='max number of epochs')

    args = vars(parser.parse_args())

    qc_tile_dir = args["qc_tile_dir"]
    fig_dst_dir = args["fig_dst_dir"]
    data_dst_dir = args["data_dst_dir"]
    rep_id = args["rep_id"]
    with_site = eval(args["with_site"])
    max_epoch = args["max_epoch"]

    all_counts_df = pd.concat([pd.read_csv(f) for f in list(pathlib.Path(qc_tile_dir).rglob("*.csv"))])
    tile_wh = np.mean([np.diff(all_counts_df[QC_X_COL]).max(), np.diff(all_counts_df[QC_Y_COL]).max()])

    prefix = f"wh_{tile_wh}_rep_{rep_id}_sites_{with_site}"
    print(prefix)

    # Create directories
    confusion_test_plot_f = os.path.join(fig_dst_dir, TEST_CONFUSION, f"{prefix}_{TEST_CONFUSION}.png")
    confusion_test_data_f = confusion_test_plot_f.replace(fig_dst_dir, data_dst_dir).replace(".png", ".csv")

    confusion_all_plot_f = os.path.join(fig_dst_dir, ALL_CONFUSION, f"{prefix}_{ALL_CONFUSION}.png")
    confusion_all_data_f = confusion_all_plot_f.replace(fig_dst_dir, data_dst_dir).replace(".png", ".csv")

    contrib_all_plot_f = os.path.join(fig_dst_dir, CONTRIBUTION, f"{prefix}_{CONTRIBUTION}.png")
    contrib_all_data_f = contrib_all_plot_f.replace(fig_dst_dir, data_dst_dir).replace(".png", ".csv")

    train_stats_plot_f = os.path.join(fig_dst_dir, TRAINING, f"{prefix}_{TRAINING}.png")
    train_stats_data_f = train_stats_plot_f.replace(fig_dst_dir, data_dst_dir).replace(".png", ".csv")

    summary_plot_f = os.path.join(fig_dst_dir, SUMMARY, f"{prefix}_{SUMMARY}.png")
    summary_data_f = summary_plot_f.replace(fig_dst_dir, data_dst_dir).replace(".png", ".csv")

    for f in [confusion_test_plot_f, confusion_test_data_f,
              confusion_all_plot_f, confusion_all_data_f,
              contrib_all_plot_f, contrib_all_data_f,
              train_stats_plot_f, train_stats_data_f,
              summary_plot_f, summary_data_f
              ]:
        d = os.path.split(f)[0]
        pathlib.Path(d).mkdir(exist_ok=True, parents=True)


    label_encoder = LabelEncoder()
    all_counts_df = combine_hypoxic(all_counts_df)

    all_counts_df[QC_NUM_LBL] = label_encoder.fit_transform(all_counts_df[QC_STR_LBL])
    exclude_cell_cols = ['Unclassified TME', 'Hypoxic Unclassified TME', "TERT C228T wt"]
    exclude_info_cols = ["label_id", "label"]
    if not with_site:
        exclude_info_cols.append(QC_SITE_COL)
    all_feature_cols = [x for x in CELL_COLS if not x in exclude_cell_cols and x in list(all_counts_df)]
    info_cols = [x for x in list(all_counts_df) if x not in CELL_COLS if not x in exclude_cell_cols if not x in exclude_info_cols]

    _features = all_counts_df[all_feature_cols].values
    _labels = all_counts_df[QC_NUM_LBL].values

    features, labels, non_empty_idx = remove_empty_quadrats(_features, _labels, 0)
    str_labels = label_encoder.inverse_transform(labels)
    scaler = MinMaxScaler()

    feature_df = pd.DataFrame(features, columns=all_feature_cols)
    info_df = all_counts_df[info_cols].iloc[non_empty_idx]

    assert feature_df.shape[0] == info_df.shape[0]

    # Add site as feature
    cat_cols = None
    if with_site:
        site_labeler = OneHotEncoder()
        site_labels = site_labeler.fit_transform(all_counts_df[QC_SITE_COL].iloc[non_empty_idx].values.reshape(-1, 1)).todense()
        site_df = pd.DataFrame(site_labels.astype(int), columns=site_labeler.categories_[0])
        cat_cols = list(site_df)
        feature_df = site_df.join(feature_df)


    features_st = scaler.fit_transform(feature_df) # Do scaling after separating?
    feature_df = pd.DataFrame(features_st, columns=list(feature_df))

    balanced_freq = 1/len(np.unique(labels))

    balance_training=True
    min_freq = np.max(calc_label_freq(labels))
    x_train_df, y_train_np, x_test_df, y_test_np, x_val_df, y_val_np = get_test_train_split(feature_df, labels,
                                                                                            balance_training=balance_training,
                                                                                            test_size=0.1,
                                                                                            val_size=0.1,
                                                                                            min_freq=min_freq,
                                                                                            max_freq=min_freq,
                                                                                            cat_cols=cat_cols)


    x_train_np = x_train_df.values
    x_test_np = x_test_df.values
    x_val_np = x_val_df.values

    y_df = pd.DataFrame({
        "split": [*["train"]*y_train_np.size, *["test"]*y_test_np.size, *["val"]*y_val_np.size],
        "label": [*[label_encoder.classes_[i] for i in y_train_np],
                *[label_encoder.classes_[i] for i in y_test_np],
                *[label_encoder.classes_[i] for i in y_val_np]
                ]
    })

    batch_size = calc_batch_size(x_train_np.shape[0], target_batch_size=2**7)
    n_batches = np.ceil(x_train_np.shape[0]/batch_size)

    train_data = numpy2torch_loader(x_train_np, y_train_np, batch_size=batch_size, shuffle=True, oversample=False)
    val_data = numpy2torch_loader(x_val_np, y_val_np, batch_size=x_val_np.shape[0], shuffle=False)
    test_data = numpy2torch_loader(x_test_np, y_test_np, batch_size=x_test_np.shape[0], shuffle=False)


    ##TODO Re-Add regularization and Change funnelD to 0.5
    N_CLASSES = len(np.unique(labels))
    N_FEATURES = x_train_np.shape[1]

    N_HIDDEN_UNITS = int(N_FEATURES*N_CLASSES)
    N_LAYERS = 10
    FUNNEL_D = int((N_HIDDEN_UNITS/N_CLASSES))
    EXPAND_D = 0
    LEARNING_RATE = 6*(10**-4)
    REGULARIZATION = 0
    optimizer_init_kwargs={"lr":LEARNING_RATE,
                           "weight_decay":REGULARIZATION
                           }

    optimizer_cls = torch.optim.AdamW
    optimizer_init_kwargs["betas"]=[0.9, 0.999]
    activation = nn.LeakyReLU()

    cls_weights, _ = calc_cls_weights_for_unbalanced(y=train_data)
    cb_loss_fxn = get_cb_loss_fxn(y=train_data)
    model = TorchClassifier(n_features=N_FEATURES, n_hidden_units=N_HIDDEN_UNITS,
                        n_layers=N_LAYERS, n_cls=N_CLASSES,
                        activation_fxn=activation,
                        optimizer_cls=optimizer_cls,
                        optimizer_init_kwargs=optimizer_init_kwargs,
                        funnel_s=FUNNEL_D,
                        expand_s=EXPAND_D,
                        cls_weights=cls_weights.type(torch.float),
                        label_smoothing=0.1,
                        loss_fxn=cb_loss_fxn,
                        dropout_p=0.15)

    epoch_num_list, train_acc_list, train_loss_list, train_roc_auc_list, val_acc_list, val_loss_list, val_roc_auc_list = model.train_model(
        train_dset=train_data,
        val_dset=val_data,
        n_epoch=max_epoch,
        minimize_val_loss=False)

    n_pts = len(epoch_num_list)
    res_df = pd.DataFrame({
        DSET_COL: [*[TRAIN_NAME]*n_pts, *[VAL_NAME]*n_pts],
        EPOCH_COL: np.hstack([epoch_num_list, epoch_num_list]),
        ROC_AUC_COL: np.hstack([train_roc_auc_list, val_roc_auc_list]),
        ACC_COL: np.hstack([train_acc_list, val_acc_list]),
        LOSS_COL: np.hstack([train_loss_list, val_loss_list])
    })

    res_df_long = res_df.melt(id_vars=[DSET_COL, EPOCH_COL], var_name=MEASURE_COL, value_name=VAL_COL)
    res_df_long = res_df_long.dropna()
    res_df_long[TILE_WH] = tile_wh
    res_df_long[REP_COL] = rep_id
    res_df_long[WITH_SITE_COL] = with_site


    g = sns.FacetGrid(res_df_long, col=MEASURE_COL, hue=DSET_COL, sharey=False)
    g.map(sns.lineplot, EPOCH_COL, VAL_COL)
    ax1, ax2, ax3 = g.axes[0]
    ax1.axvline(model.best_epoch, c="black",  linestyle='dashed')
    ax2.axvline(model.best_epoch, c="black",  linestyle='dashed')
    ax3.axvline(model.best_epoch, c="black",  linestyle='dashed')
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)
    ax3.set_ylim(0, res_df[LOSS_COL].max())
    plt.legend()
    plt.savefig(train_stats_plot_f, dpi=200)
    plt.close()

    res_df_long.to_csv(train_stats_data_f, index=False)

    model.load_model("overall")
    with torch.inference_mode():
        _, test_probs, test_pred_y = model.predict(test_data.dataset.tensors[0])

    test_pred_y = test_pred_y.detach().numpy()
    test_true_y = test_data.dataset.tensors[1].detach().numpy().astype(int)
    test_cmat_labels = label_encoder.inverse_transform(range(len(label_encoder.classes_)))
    test_acc, test_loss, test_roc_auc = model.calc_score(test_data)
    test_acc_plt = metrics.ConfusionMatrixDisplay.from_predictions(test_true_y, test_pred_y,
                                                                labels=None, display_labels=test_cmat_labels,
                                                                normalize="true", ax=None)

    plt.title(f"Test accuracy. ROC-AUC={test_roc_auc:.3f}")
    plt.savefig(confusion_test_plot_f, dpi=200)
    plt.close()

    test_confusion_mat = metrics.confusion_matrix(test_true_y, test_pred_y,
                                                  labels=None,
                                                  normalize="true")

    test_confusion_df = pd.DataFrame(test_confusion_mat, columns=test_cmat_labels, index=test_cmat_labels)
    test_confusion_long = pd.melt(test_confusion_df.reset_index(), id_vars=['index'], value_vars=test_cmat_labels, var_name='Pred', value_name='Acc').rename(mapper={"index":"True"}, axis=1)
    test_confusion_long[DSET_COL] = TEST_NAME
    test_confusion_long[WH_COL] = tile_wh
    test_confusion_long[REP_COL] = rep_id
    test_confusion_long[WITH_SITE_COL] = with_site
    test_confusion_long.to_csv(confusion_test_data_f, index=False)

    all_x = torch.from_numpy(features_st).type(torch.float)
    with torch.inference_mode():
        all_logits, all_probs, pred_y = model.predict(all_x)

    pred_y = pred_y.detach().numpy()
    true_y = labels
    cmat_labels = label_encoder.inverse_transform(range(len(label_encoder.classes_)))
    all_roc_auc = roc_auc_score(true_y, all_probs.detach().numpy(), multi_class="ovr")
    acc_plt = metrics.ConfusionMatrixDisplay.from_predictions(true_y, pred_y,
                                                                labels=None, display_labels=cmat_labels,
                                                                normalize="true", ax=None)

    plt.title(f"All accuracy. ROC-AUC = {all_roc_auc:.3f} ")
    plt.savefig(confusion_all_plot_f, dpi=200)
    plt.close()

    all_confusion_mat = metrics.confusion_matrix(true_y, pred_y,
                                                 labels=None,
                                                 normalize="true")
    all_confusion_df = pd.DataFrame(all_confusion_mat, columns=cmat_labels, index=cmat_labels)
    all_confusion_long = pd.melt(all_confusion_df.reset_index(), id_vars=['index'], value_vars=cmat_labels, var_name='Pred', value_name='Acc').rename(mapper={"index":"True"}, axis=1)
    all_confusion_long[DSET_COL] = ALL_NAME
    all_confusion_long[WH_COL] = tile_wh
    all_confusion_long[REP_COL] = rep_id
    all_confusion_long[WITH_SITE_COL] = with_site
    all_confusion_long.to_csv(confusion_all_data_f, index=False)

    val_acc, val_loss, val_roc_auc = model.calc_score(val_data)
    train_acc, train_loss, train_roc_auc = model.calc_score(train_data)

    all_acc = len(np.where(pred_y == true_y)[0])/len(true_y)
    all_loss = model.calc_loss(all_logits, torch.from_numpy(true_y)).item()

    summary_df = pd.DataFrame({DSET_COL:[TRAIN_NAME, VAL_NAME, TEST_NAME, ALL_NAME],
                               ACC_COL: [train_acc, val_acc, test_acc, all_acc],
                               ROC_AUC_COL: [train_roc_auc, val_roc_auc, test_roc_auc, all_roc_auc],
                               LOSS_COL: [train_loss, val_loss, test_loss, all_loss]})

    summary_df_long = summary_df.melt(id_vars=[DSET_COL], var_name=MEASURE_COL, value_name=VAL_COL)
    summary_df_long[WH_COL] = tile_wh
    summary_df_long[REP_COL] = rep_id
    summary_df_long[WITH_SITE_COL] = with_site
    summary_df_long.to_csv(summary_data_f, index=False)

    g = sns.FacetGrid(summary_df_long, col=MEASURE_COL)
    g.map(sns.barplot, DSET_COL, VAL_COL)
    plt.savefig(summary_plot_f)



    # Calculate integrated gradients
    zero_df = pd.DataFrame(np.zeros(feature_df.shape[1]).reshape(1, -1), columns=list(feature_df))
    baseline_np = scaler.transform(zero_df)
    baseline = torch.from_numpy(baseline_np).type(torch.float)

    ig = IntegratedGradients(model)
    torch_features = deepcopy(all_x)
    torch_features.requires_grad_()

    feature_cols = list(feature_df)
    assert train_data.dataset.tensors[0].shape[1] == torch_features.shape[1] == baseline.shape[1]

    with torch.inference_mode():
        _, _, pred_y = model.predict(torch_features)

    correct_idx = np.where(pred_y.detach().numpy() == true_y)[0]
    correct_info_df = info_df.iloc[correct_idx]

    all_attr = [None] * N_CLASSES
    for i in tqdm(range(N_CLASSES)):

        lbl_attributions, delta = ig.attribute(torch_features[correct_idx, :], target=i, baselines=baseline, return_convergence_delta=True, additional_forward_args=True, n_steps=200)

        correct_attributions = lbl_attributions.detach().numpy()
        lbl_attributions_df = pd.DataFrame(correct_attributions, columns=feature_cols)
        lbl_info_df = deepcopy(correct_info_df)
        lbl_info_df[REFLECTANCE_COL] = label_encoder.classes_[i]
        lbl_attributions_df = lbl_info_df.join(lbl_attributions_df)
        all_attr[i] = lbl_attributions_df

        list(lbl_attributions_df)


    attributions_df = pd.concat(all_attr)
    melt_args = {"var_name":CELL_COL_NAME, "value_name":INFLUENCE_COL_NAME, "id_vars":[*info_cols, REFLECTANCE_COL]}


    contrib_plot_df = attributions_df.melt(**melt_args)
    contrib_plot_df[WH_COL] = tile_wh
    contrib_plot_df[REP_COL] = rep_id
    contrib_plot_df.to_csv(contrib_all_data_f, index=False)

    g = sns.FacetGrid(contrib_plot_df, col=REFLECTANCE_COL)
    g.map(sns.barplot, CELL_COL_NAME, INFLUENCE_COL_NAME)
    for ax in g.axes_dict.values():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, size=6, ha='right')
    plt.savefig(contrib_all_plot_f, dpi=200)
    plt.close()