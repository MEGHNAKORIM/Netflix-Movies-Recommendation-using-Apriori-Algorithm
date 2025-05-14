
# 🎬 Netflix Movie Recommendation using Apriori Algorithm

This project implements a recommendation system that suggests movies based on users' watch history using **Association Rule Learning** with the **Apriori Algorithm**. The idea is to identify patterns like:

> “People who watched this also watched...”  
> “Recommended for you based on your viewing behavior.”

---

## 💡 About Apriori Algorithm

The **Apriori Algorithm** is used to identify frequent combinations of items in datasets and generate **association rules**. It's widely used in:

- Recommendation systems
- Market basket analysis
- Fraud detection
- Web usage mining

Apriori uses the principle:

> *If an itemset is frequent, all of its subsets are also frequent.*

---

## 📊 Metrics Used

### 🔹 Support
Indicates how frequently an item appears in the dataset.

**Example**:
- If 150 out of 1000 users watched *Movie A*,  
  `Support(Movie A) = 150 / 1000 = 0.15`

### 🔹 Confidence
Measures the likelihood of *Movie B* being watched given that *Movie A* was watched.

**Example**:
- 80 of 150 users who watched *Movie A* also watched *Movie B*,  
  `Confidence(A → B) = 80 / 150 = 0.533`

### 🔹 Lift
Evaluates how much more likely *Movie B* is watched with *Movie A* than by random chance.

**Formula**:
```text
Lift(A → B) = Confidence(A → B) / Support(B)
```
- A lift > 1 indicates a strong positive association.

---

## 📂 Dataset

- The dataset contains movie watch histories of users.
- Each row represents movies watched by one user.
- Used as transactions for the Apriori algorithm.

---

## ✅ Steps to Run

1. **Install Dependencies**:
   ```bash
   pip install pandas numpy apyori
   ```

2. **Load Dataset**:
   ```python
   import pandas as pd
   data = pd.read_csv('NetflixMoviesWatched.csv', header=None, encoding='latin-1')
   ```

3. **Preprocess Data**:
   Convert rows to a list of watched movies.

4. **Apply Apriori**:
   ```python
   from apyori import apriori
   rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, max_length=2)
   ```

5. **Analyze Rules**:
   Extract and visualize the most relevant movie pairs.

---

## 📈 Sample Output

| Movie 1       | Movie 2         | Support |
|---------------|-----------------|---------|
| Inception     | Interstellar    | 0.0048  |
| Titanic       | Avatar          | 0.0041  |
| ...           | ...             | ...     |

---

## 🔍 Use Cases

- **Streaming Services**: Movie & show recommendations
- **E-commerce**: Product bundling suggestions
- **Retail**: Market basket analysis
- **Finance**: Fraud pattern recognition

---

## 📌 Notes

- This is an **unsupervised learning** approach.
- It suggests **co-watched movies**, not ratings.
- Efficient for discovering **hidden viewing patterns**.

---

## 🤝 Contributions

Feel free to contribute, suggest improvements, or report issues!

