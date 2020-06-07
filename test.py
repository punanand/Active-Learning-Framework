from sklearn.datasets import load_digits
import random

lst = list(range(10))
lst2 = random.sample([1,2,3], 2)
lst3 = [lst[idx] for idx in lst2]
print(lst3)