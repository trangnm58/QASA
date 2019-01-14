# Quasar-T
# from converters import main
# main("QuasarTConverter", "quasar_htut")

# from build_data import main
# main("quasar_htut", "QuasarTDataset", "data/w2v_model/glove.6B.300d.txt")

# from dataset import main
# main("quasar_htut", "QuasarTDataset", "train_pairwise")
#
# from quasar_model import main
# main("quasar_no_neural_neg5", "quasart", "train_pairwise.pickle",
#      True, epoch=50, k_neg=5, is_continue=False, fast_sample=False)

# from htut_RN_ranker import main
# main("htut_RN_ranker", "quasart_glove", "train_pairwise.pickle", is_evaluate=True,
#      early_stopping=True, epoch=50, k_neg=5, is_continue=False)

from qasa import main
main("qasa_neural", "quasart", "train_pairwise.pickle", is_evaluate=True,
     early_stopping=True, epoch=50, k_neg=5, is_continue=False)

# from baseline import main
# main("baseline_neural", "quasart", "train_pairwise.pickle", is_evaluate=True,
#      early_stopping=True, epoch=50, k_neg=5, is_continue=False)
