from classifiers import BinaryClassifier, MultiClassClassifier

binary = BinaryClassifier()
binary.train()

multi = MultiClassClassifier()
multi.train()

binary.get_report()
binary.get_cmatrix()

multi.get_report()
multi.get_cmatrix()