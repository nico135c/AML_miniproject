from classifiers import BinaryClassifier, MultiClassClassifier

binary = BinaryClassifier()
multi = MultiClassClassifier()

models = [binary, multi]

for model in models:
    model.train()
    model.create_plots()