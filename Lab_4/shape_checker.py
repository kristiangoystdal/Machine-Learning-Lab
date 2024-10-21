from data_loader import load_data

Xtrain2_a, Xtrain2_b, Ytrain2_a, Ytrain2_b, Xtest2_a, Xtest2_b = load_data()

# Check the shapes
print("Xtrain2_a shape: ", Xtrain2_a.shape)
print("Xtrain2_b shape: ", Xtrain2_b.shape)
print("Ytrain2_a shape: ", Ytrain2_a.shape)
print("Ytrain2_b shape: ", Ytrain2_b.shape)
print("Xtest2_a shape: ", Xtest2_a.shape)
print("Xtest2_b shape: ", Xtest2_b.shape)
