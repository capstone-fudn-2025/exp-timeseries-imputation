# Ghost Time Series Training Modules

Official repository for time series imputation research by the AI Department at FPT University Da Nang.

---

## 📄 Papers & Publications

If you use this repository or dataset in your research, please refer to our published papers:

### 1. Main Paper
* **Title:** *Investigating 1DCNN Architectures for Improving Univariate Time Series Imputation*
* **Authors:** Thieu-Quang Dinh, Ngoc-Huy Dao, Quang-Minh Doan, Thi-Minh-Thu Le, Thi-Thu-Hong Phan
* **Conference / Book:** *International Conference on Digital Technologies and Applications (ICDTA)* / Springer Lecture Notes in Networks and Systems
* **DOI:** [10.1007/978-3-032-06061-7_6](https://doi.org/10.1007/978-3-032-06061-7_6)

### 2. Related Paper
* **Title:** *WBDI Approach for Univariate Time Series Imputation*
* **Authors:** Ngoc-Huy Dao, Quang-Minh Doan, Thieu-Quang Dinh, Quan-Bao Nguyen, Thi-Thu-Hong Phan
* **Journal:** *Informatica* (Vol 48, 2024)

---

## 📌 Citation

When using this code or referencing our work, please cite our paper:

### BibTeX
```bibtex
@inbook{Dinh2026,
  author    = {Dinh, Thieu-Quang and Dao, Ngoc-Huy and Doan, Quang-Minh and Le, Thi-Minh-Thu and Phan, Thi-Thu-Hong},
  title     = {Investigating 1DCNN Architectures for Improving Univariate Time Series Imputation},
  booktitle = {Digital Technologies and Applications},
  year      = {2026},
  publisher = {Springer Nature Switzerland},
  address   = {Cham},
  doi       = {10.1007/978-3-032-06061-7_6},
  url       = {https://doi.org/10.1007/978-3-032-06061-7_6}
}
```

### Standard Citation
> T.-Q. Dinh, N.-H. Dao, Q.-M. Doan, T.-M.-T. Le, and T.-T.-H. Phan, "Investigating 1DCNN Architectures for Improving Univariate Time Series Imputation," in *Digital Technologies and Applications*, Springer, 2026, pp. 71–81. DOI: [10.1007/978-3-032-06061-7_6](https://doi.org/10.1007/978-3-032-06061-7_6).

---

## 💻 Code & Usage

### ⚡ Model Aliases

- `lr` : Linear Regression
- `knn` : K-Nearest Neighbor
- `svm` : Support Vector Machine
- `dt` : Decision Tree
- `et` : Extra Tree
- `ada` : AdaBoost
- `bag` : Bagging
- `gb` : Gradient Boosting
- `rf` : Random Forest
- `xgb` : XGBoost
- `vote` : Voting of Machine Learning
- `jeong` : Jeong Stacking
- `arima` : AutoRegressive Integrated Moving Average
- `rnn` : Recurrent Neural Network
- `lstm` : Long-Short Term Memory
- `cnn1d` : Convolution Network 1D
- `gans` : Generative Adversarial Network
- `transformer` : Transformer
- `kan`: Kolmogorov-Arnold Network

---

## 📖 Documentation

### Training on Local

- Clone project

```bash
# Clone project
git clone https://github.com/Ming-doan/timeseries-imputation.git timeseries
# Change directory to project
cd timeseries
# Install dependencies
pip install -r requirements.txt
```

- Select model to train in notebook `ghost.ipynb`:

```python
# Add the model alias on 👇 this list
MODELS = get_by_aliases(['rf', 'svm'], rf_n_estimators=100)
# Add model custom parameters .........☝️. Start with prefix of model alias.
```

- Add your own model. Create a new file in `modules/models/your_model.py`:

```python
from ._base import BaseModelWrapper

# Define class of your custom model
class YourModel(BaseModelWrapper):
    # Define the name of your model. This name will display during training or plotting results.
    name = "MyCustomModel"

    def __init__(self, **kwargs):
        self.is_generate = False
        # Define your arguments here
        ...

    def fit(self, generator, x, y):
        # If `is_generate` is False, this method will receive x, y as `np.NDArray`.
        # If `is_generate` is True, this method will receive generator as `WindowGenerator`.
        ...

    def predict(self, generator, x):
        # If `is_generate` is False, this method will receive x as `np.NDArray`.
        # If `is_generate` is True, this method will receive generator as `WindowGenerator`.
        ...

    def forecast(self, x, steps):
        # The forecast function is used to forecast the future values.
        # The x is the last window of data as `np.NDArray`.
        # The steps is how many data points to generate.
        ...

    def summary(self):
        # Show the summary of your model
        ...

    def reset(self):
        # Reset the parameters of your model for a new dataset.
        ...
```

- Add to `modules/models/__init__.py` and `modules/alias.py` for model aliases:

```python
# Adding to `modules/models/__init__.py` for easy importing
from .your_model import YourModel

# Adding to `modules/alias.py` for creating the alias for your model
def get_by_alias(alias, **kwargs):
    ...
    if alias == 'your_defined_alias':
        return YourModel(**filter_and_format('your_defined_alias', kwargs))
    ...
```

- Using `WindowGenerator`: Dynamic generator for creating time series datasets with window size and batch size.

```python
generator = WindowGenerator(df, WINDOW_SIZE, BATCH_SIZE)
# Using slice index to get the batch of data split by window.
generator[0] # The output shape is (batch_size, window_size, n_features)

# Using `generate()` to generate full data without using batches.
generator.generate() # The output shape is (data_length, window_size, n_features)
```

```python
# Iterate through data with batch size
for data_with_batch in generator:
    print(data_with_batch.shape) # Output: (batch_size, window_size, n_features)
```

- Define custom preprocessing:

```python
from modules.utils.preprocessing import Plugins

# Create class of custom preprocessing
class YourPreprocessing(Plugins):
    def __init__(self): ...

    # Define your algorithm for preprocessing (`x` is `np.NDArray`)
    def flow(self, x): ...

    # Define your algorithm for reversing back to original (`x` is `np.NDArray`)
    def reverse_flow(self, x): ...
```

```python
# Add your preprocessing into use
preprocessor = Preprocessing()
preprocessor.add_plugin(YourPreprocessing())
preprocessor.flow(dataframe)
```

- Custom Training Callbacks:

```python
from modules.utils.callbacks import Callback

# Define your custom callback
class YourCallback(Callback):
    def __init__(self):
        super().__init__()

    # Algorithm after model predicting results (`y_true` and `y_pred` are `np.NDArray`)
    def after_predict(self, y_true, y_pred): ...

    # Algorithm after model forecasting results (`y_true` and `y_fore` are `np.NDArray`)
    def after_forecast(self, y_true, y_fore): ...
```

```python
# Using callback
your_callback = YourCallback()
trainer.train(generator, callbacks=[your_callback])
```

### Training on Colab

👉 [Phantom Colab Notebook](https://colab.research.google.com/drive/1hokWxs8VnsdT_CMmTas-qHiO9KuARAVx?usp=sharing)

### Training with Docker

- Build Docker image (make sure **Docker** is installed):

```sh
docker build -t ts-cullinan .
```

- Start Docker desktop or daemon.
- Config and run file `spectre.py`.

---

## 🕹️ Contributors

* **Đào Ngọc Huy** ([Facebook](https://www.facebook.com/tonydnh43))
* **Đoàn Quang Minh** ([Facebook](https://www.facebook.com/ming.doan/))
* **Đinh Thiều Quang** ([Facebook](https://www.facebook.com/quang.dinh.90813236))
* **Nguyễn Quân Bảo** ([Facebook](https://www.facebook.com/profile.php?id=100037350121063))
* **Lê Thị Minh Thư** ([Facebook](https://www.facebook.com/minhthu.34))
