import numpy as np
from numpy.linalg import svd, inv, pinv
import scipy.linalg
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from scipy.linalg import ldl
import time
from aeon.datasets.tsc_datasets import multivariate, univariate
from aeon.datasets import load_classification
from tqdm import tqdm



def generate_Wr(Nx, alpha_r, beta_r):
    np.random.seed(0)
    Wr = np.zeros(Nx * Nx)
    Wr[0:int(Nx * Nx * beta_r)] = np.random.normal(0, 1 / np.sqrt(beta_r * Nx), int(Nx * Nx * beta_r))
    np.random.shuffle(Wr)
    Wr = Wr.reshape((Nx, Nx))
    Wr *= alpha_r
    return Wr

def generate_Wb(Nx, Ny, alpha_b, beta_b):
    np.random.seed(0)
    Wb = np.zeros(Nx * Ny)
    Wb[0:int(Nx * Ny * beta_b)] = np.random.uniform(-1, 1, int(Nx * Ny * beta_b)) # beta_b = 非ゼロ率
    np.random.shuffle(Wb)
    Wb = Wb.reshape((Nx, Ny))
    Wb *= alpha_b
    return Wb

def generate_Wi(Nx, Ni, alpha_i, beta_i):
    np.random.seed(0)
    Wi = np.zeros(Nx * Ni)
    Wi[0:int(Nx * Ni * beta_i)] = np.random.uniform(-1, 1, int(Nx * Ni * beta_i)) # beta_b = 非ゼロ率
    np.random.shuffle(Wi)
    Wi = Wi.reshape((Nx, Ni))
    Wi *= alpha_i
    return Wi

def generate_Wi_separated(Nx, Ni, alpha_i, beta_i):
    np.random.seed(0)
    Wi = np.zeros((Nx, Ni))
    Wi_ = np.zeros(int(Nx / Ni))
    for i in range(Ni):
        Wi_[:int(beta_i * Nx / Ni)] = np.random.uniform(-1, 1, int(beta_i * Nx / Ni))
        np.random.shuffle(Wi_)
        Wi[i * int(Nx / Ni): (i + 1) * int(Nx / Ni), i] = Wi_
    np.random.shuffle(Wi)
    Wi *= alpha_i
    return Wi

def fx(x):
    return np.tanh(x)
    # return x * (x > 0) # ReLU
    # return (1 + np.tanh((1/2) * x)) / 2

def f_kernel(x):
    return x
    # return np.tanh(x)

def ridge_regression(X, Y, alpha):
    W = np.linalg.inv(X.T @ X + alpha * np.eye(X.shape[1])) @ X.T @ Y
    return W


def preprocess_data(raw_data):
    train_data = raw_data['x_train']
    test_data = raw_data['x_test']
    train_data = (train_data / 5.0) + 1.0
    test_data = (test_data / 5.0) + 1.0

    len_train_data = train_data.shape[1]
    len_test_data = test_data.shape[1]
    len_min = min([len_train_data, len_test_data])
    train_data = train_data[:, :len_min, :]
    test_data = test_data[:, :len_min, :]

    return train_data, test_data


class Reservoir:
    def __init__(self, param):
        self.T1 = param['T1']
        self.Nx = param['Nx']
        self.Ny = param['Ny']
        self.Ni = param['Ni']
        self.N_P = param['N_P']
        self.label_size = param['label_size']
        self.num_train = param['num_train']

        self.tau = param['tau']
        self.dt = param['dt']
        self.sigma_noise = param['sigma_noise']
        self.alpha = param['alpha']

        self.alpha_r = param['alpha_r']
        self.beta_r = param['beta_r']
        self.alpha_i = param['alpha_i']
        self.beta_i = param['beta_i']
        self.alpha_b = param['alpha_b']
        self.beta_b = param['beta_b']

        sub_array = np.identity(self.N_P)
        self.P = np.array([sub_array] * self.label_size)
        self.cov = np.zeros_like(self.P)
        sub_vector = np.zeros(self.N_P)
        self.mu = np.array([sub_vector] * self.label_size)

        self.svm = SVC(kernel='linear')
        gamma = 0.01
        self.svm_rbf = SVC(kernel='rbf', gamma=gamma, C=1.0)


    def generate_weight_matrices(self):
        self.Wr = generate_Wr(self.Nx, self.alpha_r, self.beta_r)
        self.Wb = generate_Wb(self.Nx, self.Ny, self.alpha_b, self.beta_b)
        self.Wi = generate_Wi(self.Nx, self.Ni, self.alpha_i, self.beta_i)
        # self.Wi = generate_Wi_separated(self.Nx, self.Ni, self.alpha_i, self.beta_i)
        self.Wo = np.zeros((self.Ny, self.N_P))

    def load_weight_matrices(self, Wr, Wb, Wi):
        self.Wr = Wr * alpha_r
        self.Wb = Wb * alpha_b
        self.Wi = Wi * alpha_i
        self.Wo = np.zeros((self.Ny, self.N_P))
        self.Wi[Wi_zero_idx, 0] = 0
        self.Wb[Wb_zero_idx, 0] = 0

    def reset(self, x_init=0.2):
        self.X = np.zeros((self.T1, self.Nx))
        self.Y = np.zeros((self.T1, self.Ny))
        # self.X[0, :] = np.random.uniform(-1, 1, self.Nx) * 0.2
        self.X[0, :] = x_init
        self.Y[0, :] = np.zeros(self.Ny)

    def freerun(self):
        self.reset()
        self.T_free = 10000
        self.X_ = np.zeros((self.T_free, self.Nx))
        self.Y_ = np.zeros((self.T_free, self.Ny))
        self.X_[0] = self.X[0]
        for n in range(self.T_free):
            self.run_one_step(n, np.zeros(self.Ni), 'freerun')
        self.x_init = self.X_[self.T_free - 1]

    def run_one_step(self, n, input, mode):
        if mode == 'freerun':
            x = self.X_[n, :]
            y = self.Y_[n, :]
        else:
            x = self.X[n, :]
            y = self.Y[n, :]

        sum = np.zeros(self.Nx)
        sum += self.Wr @ fx(x)
        sum += self.Wi @ input

        ## Dynamics of learky-integrator
        dx = (self.dt / self.tau) * (- x + sum)
        x += dx
        y = self.Wo @ fx(x[:reservoir.N_P])

        if mode == 'freerun':
            if n < self.T_free - 1:
                self.X_[n + 1, :] = x
                self.Y_[n + 1, :] = y
        else:
            if n < self.T1 - 1:
                self.X[n + 1, :] = x
                self.Y[n + 1, :] = y

def train_MDRS(train_data, idx_labels, reservoir):
    sub_array = np.identity(reservoir.N_P)
    reservoir.P = np.array([sub_array] * reservoir.label_size)
    sub_vector = np.zeros(reservoir.N_P)
    reservoir.mu = np.array([sub_vector] * reservoir.label_size)


    train_length = np.zeros(reservoir.label_size)

    with tqdm(total=reservoir.label_size, desc="Training MDRS", unit="class") as pbar:
        for label in range(reservoir.label_size):
            X_list = []
            for idx in idx_labels['train'][label]:
                data = train_data[idx]
                reservoir.reset(reservoir.x_init)
                for n in range(reservoir.T1):
                    train_length[label] += 1
                    reservoir.run_one_step(n, data[n], 'test')
                X_list.append(reservoir.X[:, :reservoir.N_P].copy())

            X_concat = np.vstack(X_list)
            reservoir.mu[label] = np.mean(X_concat, axis=0)
            cov_mat = np.cov(X_concat, rowvar=False) + reservoir.alpha * np.identity(reservoir.N_P)
            reservoir.P[label] = np.linalg.inv(cov_mat)
            reservoir.cov[label] = cov_mat

            pbar.update(1)

def train_readout(train_data, idx_labels, reservoir, alpha=None):
    num_data = reservoir.num_train
    X_bar_list = np.empty((num_data, reservoir.N_P * reservoir.T1))
    X_list = np.empty((num_data * reservoir.T1, reservoir.N_P))
    Y_list = np.empty((num_data * reservoir.T1, reservoir.Ny))
    label_list = []
    label_bar_list = []
    count = 0
    count_ = 0
    train_length = np.zeros(reservoir.label_size)
    eye = np.eye(reservoir.label_size)

    with tqdm(total=reservoir.label_size, desc="Training Readout", unit="class") as pbar:
        for label in range(reservoir.label_size):
            for idx in idx_labels['train'][label]:
                data = train_data[idx]
                reservoir.reset(reservoir.x_init)
                for n in range(reservoir.T1):
                    train_length[label] += 1
                    reservoir.run_one_step(n, data[n], 'test')
                    X_list[count_] = fx(reservoir.X[n, :reservoir.N_P])
                    Y_list[count_] = eye[label]
                    label_list.append(label)
                    count_ += 1
                X_bar_list[count] = reservoir.X[:, :reservoir.N_P].flatten()
                label_bar_list.append(label)
                count += 1
            pbar.update(1)

    if alpha == None:
        reservoir.Wo = Y_list.T @ np.linalg.pinv(X_list.T)
    else:
        reservoir.Wo = ridge_regression(X_list, Y_list, alpha).T

    return None

def train_SVM(train_data, idx_labels, reservoir):
    num_data = reservoir.num_train
    X_bar_list = np.empty((num_data, reservoir.N_P * reservoir.T1))
    X_list = np.empty((num_data * reservoir.T1, reservoir.N_P))
    Y_list = np.empty((num_data * reservoir.T1, reservoir.Ny))
    label_list = []
    label_bar_list = []
    count = 0
    count_ = 0
    train_length = np.zeros(reservoir.label_size)
    eye = np.eye(reservoir.label_size)
    for label in range(reservoir.label_size):
        for idx in idx_labels['train'][label]:
            data = train_data[idx]
            reservoir.reset(reservoir.x_init)
            for n in range(reservoir.T1):
                train_length[label] += 1
                reservoir.run_one_step(n, data[n], 'test')
                X_list[count_] = fx(reservoir.X[n, :reservoir.N_P])
                Y_list[count_] = eye[label]
                label_list.append(label)
                count_ += 1

            X_bar_list[count] = reservoir.X[:, :reservoir.N_P].flatten()
            label_bar_list.append(label)
            count += 1
    reservoir.svm.fit(X_bar_list, label_bar_list)
    return None

def train_SVM_RBF(train_data, idx_labels, reservoir):
    num_data = reservoir.num_train
    X_bar_list = np.empty((num_data, reservoir.N_P * reservoir.T1))  # データ数×リザバー時系列の行列
    X_list = np.empty((num_data * reservoir.T1, reservoir.N_P))
    Y_list = np.empty((num_data * reservoir.T1, reservoir.Ny))
    label_list = []
    label_bar_list = []
    count = 0
    count_ = 0
    train_length = np.zeros(reservoir.label_size)
    eye = np.eye(reservoir.label_size)
    for label in range(reservoir.label_size):
        for idx in idx_labels['train'][label]:
            data = train_data[idx]
            reservoir.reset(reservoir.x_init)
            for n in range(reservoir.T1):
                train_length[label] += 1
                reservoir.run_one_step(n, data[n], 'test')
                X_list[count_] = fx(reservoir.X[n, :reservoir.N_P])
                Y_list[count_] = eye[label]
                label_list.append(label)
                count_ += 1
            X_bar_list[count] = reservoir.X[:, :reservoir.N_P].flatten()
            label_bar_list.append(label)
            count += 1
    reservoir.svm_rbf.fit(X_bar_list, label_bar_list)
    return None


def classify_MDRS(data, correct_label, reservoir, MDRS_mode='mean'):
    interval = 1
    sub_array = np.empty(int(reservoir.T1 / interval))
    MDRS = np.array([sub_array] * reservoir.label_size)
    MDRS_mod = np.array([sub_array] * reservoir.label_size)
    reservoir.reset(reservoir.x_init)
    for n in range(reservoir.T1):
        if n < data.shape[0]:
            reservoir.run_one_step(n, data[n], 'test')
            for label in range(reservoir.label_size):
                if n % interval == 0:
                    x = reservoir.X[n, :reservoir.N_P]
                    MDRS[label][int(n / interval)] = (f_kernel(x) - reservoir.mu[label]).T @ reservoir.P[label] @ (
                            f_kernel(x) - reservoir.mu[label])
        elif n >= data.shape[0]:
            reservoir.run_one_step(n, np.ones_like(data[0]), 'test')
        else:
            print('break time:', n)
            break

    mean_MDRS = np.empty(reservoir.label_size)
    max_MDRS = np.empty(reservoir.label_size)
    KL_MDRS = np.empty(reservoir.label_size)

    for label in range(reservoir.label_size):
        mean_MDRS[label] = np.mean(MDRS[label])
        max_MDRS[label] = np.max(MDRS[label])

        lu, d, perm = ldl(reservoir.cov[label], lower=True)
        log_det_P = - np.sum(np.log(np.diag(d)))

        KL_MDRS[label] = mean_MDRS[label] - log_det_P

        if MDRS_mode == 'mean':
            MDRS_mod[label] = MDRS[label]
        elif MDRS_mode == 'KL':
            MDRS_mod[label] = MDRS[label] - log_det_P

    if MDRS_mode == 'mean':
        inferred_label = np.argmin(mean_MDRS)
    elif MDRS_mode == 'max':
        inferred_label = np.argmin(max_MDRS)
    elif MDRS_mode == 'KL':
        inferred_label = np.argmin(KL_MDRS)

    return correct_label, inferred_label, MDRS_mod.T

def classify_readout(data, correct_label, reservoir):
    reservoir.reset(reservoir.x_init)
    reservoir_output = np.empty((reservoir.T1, reservoir.label_size))
    for n in range(reservoir.T1):
        if n < data.shape[0]:
            reservoir.run_one_step(n, data[n], 'test')
        elif n >= data.shape[0]:
            reservoir.run_one_step(n, np.ones_like(data[0]), 'test')
        reservoir_output[n] = reservoir.Wo @ fx(reservoir.X[n, :reservoir.N_P])

    X_bar = np.mean(fx(reservoir.X[:, :reservoir.N_P]), axis=0)
    Y_bar = reservoir.Wo @ X_bar

    inferred_label = np.argmax(Y_bar)

    return correct_label, inferred_label, reservoir_output

def classify_SVM(data, correct_label, reservoir):
    reservoir.reset(reservoir.x_init)
    for n in range(reservoir.T1):
        if n < data.shape[0]:
            reservoir.run_one_step(n, data[n], 'test')
        elif n >= data.shape[0]:
            reservoir.run_one_step(n, np.ones_like(data[0]), 'test')

    X_bar = reservoir.X[:data.shape[0], :reservoir.N_P].flatten()
    inferred_label = reservoir.svm.predict(X_bar.reshape(1, -1))[0]

    return correct_label, inferred_label

def classify_SVM_RBF(data, correct_label, reservoir):
    reservoir.reset(reservoir.x_init)
    for n in range(reservoir.T1):
        if n < data.shape[0]:
            reservoir.run_one_step(n, data[n], 'test')
        elif n >= data.shape[0]:
            reservoir.run_one_step(n, np.ones_like(data[0]), 'test')

    X_bar = reservoir.X[:data.shape[0], :reservoir.N_P].flatten()
    inferred_label = reservoir.svm_rbf.predict(X_bar.reshape(1, -1))[0]

    return correct_label, inferred_label


def evaluate_model(benchmark, test_data, idx_labels, reservoir, method, method_mapping):

    correct_labels = []
    inferred_labels = []
    outputs_array = np.empty((test_data.shape[0], reservoir.T1, reservoir.label_size))

    with tqdm(total=reservoir.label_size, desc="Testing", unit="class") as pbar:
        for label in range(reservoir.label_size):
            for i in idx_labels['test'][label]:
                if method == 'readout':
                    correct_label, inferred_label, outputs_array[i] = classify_readout(test_data[i], label, reservoir)
                elif method == 'SVM':
                    correct_label, inferred_label = classify_SVM(test_data[i], label, reservoir)
                    outputs_array[i] = None
                elif method == 'SVM_RBF':
                    correct_label, inferred_label = classify_SVM_RBF(test_data[i], label, reservoir)
                    outputs_array[i] = None
                elif method == 'MDRS':
                    correct_label, inferred_label, outputs_array[i] = classify_MDRS(test_data[i], label, reservoir, MDRS_mode='mean')
                elif method == 'MDRS_KL':
                    correct_label, inferred_label, outputs_array[i] = classify_MDRS(test_data[i], label, reservoir, MDRS_mode='KL')
                correct_labels.append(correct_label)
                inferred_labels.append(inferred_label)
            pbar.update(1)

    if method in ["readout", "MDRS_KL", "MDRS"]:
        np.savez_compressed(f'results/{method_mapping[method]}_{benchmark}.npz', data=outputs_array)

    cm = confusion_matrix(correct_labels, inferred_labels)

    accuracy = cm.trace() / cm.sum()

    average = 'macro'
    precision = precision_score(correct_labels, inferred_labels, average=average)
    recall = recall_score(correct_labels, inferred_labels, average=average)
    f1 = f1_score(correct_labels, inferred_labels, average=average)

    return cm, accuracy, precision, recall, f1


if __name__ == "__main__":
    for benchmark in multivariate:
    # for benchmark in univariate:
            print('Benchmark:',benchmark)

            print('Loading train data...')
            X_train, y_train, meta = load_classification(benchmark, split="train", return_metadata=True)
            print('Loading test data...')
            X_test, y_test = load_classification(benchmark, split="test")

            data_path = f'dataset/{benchmark}'
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            np.save(f'{data_path}/X_train.npy', X_train)
            np.save(f'{data_path}/y_train.npy', y_train)
            np.save(f'{data_path}/X_test.npy', X_test)
            np.save(f'{data_path}/y_test.npy', y_test)
            np.savez(f'{data_path}/meta.npz', **meta)

            # X_train = np.load(f'dataset/{benchmark}/X_train.npy')
            # y_train = np.load(f'dataset/{benchmark}/y_train.npy')
            # X_test = np.load(f'dataset/{benchmark}/X_test.npy')
            # y_test = np.load(f'dataset/{benchmark}/y_test.npy')
            # meta = np.load(f'dataset/{benchmark}/meta.npz')

            classes = meta['class_values']
            idx_labels = {"train": {}, "test": {}}

            for label, cls in enumerate(classes):
                idx_labels["train"][label] = np.where(y_train == cls)[0]
                idx_labels["test"][label] = np.where(y_test == cls)[0]

            raw_data = {}
            raw_data['x_train'] = X_train.transpose(0, 2, 1)
            raw_data['x_test'] = X_test.transpose(0, 2, 1)

            train_data, test_data = preprocess_data(raw_data)

            param = {}

            param['T1'] = train_data.shape[1]
            param['Nx'] = 500
            param['Ni'] = train_data.shape[2]
            param['Ny'] = len(classes)
            param['N_P'] = 200 # subsampling size
            param['label_size'] = len(classes)
            param['num_train'] = len(y_train)

            mix_ratio = 0.9
            tau_small = 0.01
            tau_large = 0.1
            param['tau'] = np.zeros(param['Nx'])
            param['tau'][:int(param['Nx'] * (1 - mix_ratio))] = tau_small
            param['tau'][int(param['Nx'] * (1 - mix_ratio)):] = tau_large

            param['dt'] = 0.01
            param['sigma_noise'] = 0
            param['alpha'] = 0.00001 # regularization coefficient


            param['alpha_r'] = 1.5
            param['beta_r'] = 1.0
            param['alpha_i'] = 1.0
            param['beta_i'] = 0.25
            param['alpha_b'] = 1.0
            param['beta_b'] = 0.25

            reservoir = Reservoir(param)
            reservoir.generate_weight_matrices()

            reservoir.freerun()

            methods = ["readout", "MDRS_KL", "SVM", "SVM_RBF"]
            method_mapping = {
                "readout": "Regression",
                "MDRS_KL": "GRSA",
                "SVM": "SVM_linear",
                "SVM_RBF": "SVM_RBF"
            }

            metrics_data = {
                "Method": [],
                "Accuracy": [],
                "Precision": [],
                "Recall": [],
                "F1 Score": [],
                "Training Time": [],
                "Inference Time": [],
                "Total Time": []
            }
            confusion_matrices = {}

            for method in methods:
                start_train = time.time()

                if method == "readout":
                    train_readout(train_data, idx_labels, reservoir)
                elif method == "MDRS_KL":
                    train_MDRS(train_data, idx_labels, reservoir)
                elif method == "MDRS":
                    train_MDRS(train_data, idx_labels, reservoir)
                elif method == "SVM":
                    train_SVM(train_data, idx_labels, reservoir)
                elif method == "SVM_RBF":
                    train_SVM_RBF(train_data, idx_labels, reservoir)

                end_train = time.time()

                start_infer = time.time()

                cm_test, acc_test, prec_test, rec_test, f1_test = evaluate_model(benchmark, test_data, idx_labels, reservoir, method, method_mapping)

                end_infer = time.time()

                train_time = end_train - start_train
                infer_time = end_infer - start_infer
                total_time = train_time + infer_time

                metrics_data["Method"].append(method_mapping[method])
                metrics_data["Training Time"].append(train_time)
                metrics_data["Inference Time"].append(infer_time)
                metrics_data["Total Time"].append(total_time)
                metrics_data["Accuracy"].append(acc_test)
                metrics_data["Precision"].append(prec_test)
                metrics_data["Recall"].append(rec_test)
                metrics_data["F1 Score"].append(f1_test)

                confusion_matrices[method] = cm_test

            metrics_df = pd.DataFrame(metrics_data)

            analysis_path = f'analysis/{benchmark}'
            if not os.path.exists(analysis_path):
                os.makedirs(analysis_path)

            metrics_csv_path = os.path.join(analysis_path, "performance_metrics.csv")
            metrics_df.to_csv(metrics_csv_path, index=False)
            print(f"Training and inference metrics saved to {metrics_csv_path}")

            for method, cm in confusion_matrices.items():
                cm_path = os.path.join(analysis_path, f"cm_{method_mapping[method]}.csv")
                pd.DataFrame(cm).to_csv(cm_path, index=True)
                print(f"Confusion matrix for {method} saved to {cm_path}")
