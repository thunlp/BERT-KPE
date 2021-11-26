def get_class(name):
    if name == "bert2span":
        return (
            batchify_bert2span_features_for_train,
            batchify_bert2span_features_for_test,
        )
    elif name == "bert2tag":
        return batchify_bert2tag_features_for_train, batchify_bert2tag_features_for_test
    elif name == "bert2chunk":
        return (
            batchify_bert2chunk_features_for_train,
            batchify_bert2chunk_features_for_test,
        )
    elif name == "bert2rank":
        return (
            batchify_bert2rank_features_for_train,
            batchify_bert2rank_features_for_test,
        )
    elif name == "bert2joint":
        return (
            batchify_bert2joint_features_for_train,
            batchify_bert2joint_features_for_test,
        )
    raise RuntimeError("Invalid retriever class: %s" % name)


from .loader_utils import build_dataset
from .bert2span_dataloader import (
    batchify_bert2span_features_for_train,
    batchify_bert2span_features_for_test,
)
from .bert2tag_dataloader import (
    batchify_bert2tag_features_for_train,
    batchify_bert2tag_features_for_test,
)
from .bert2chunk_dataloader import (
    batchify_bert2chunk_features_for_train,
    batchify_bert2chunk_features_for_test,
)
from .bert2rank_dataloader import (
    batchify_bert2rank_features_for_train,
    batchify_bert2rank_features_for_test,
)
from .bert2joint_dataloader import (
    batchify_bert2joint_features_for_train,
    batchify_bert2joint_features_for_test,
)
