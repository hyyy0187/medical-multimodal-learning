
import torch
import torch.nn.functional as F


def clip_loss(
    image_embeddings,
    text_embeddings,
    temperature=0.07,
    uids=None
):
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    batch_size = image_embeddings.size(0)

    if text_embeddings.size(0) != batch_size:
        raise ValueError("Image/text batch sizes differ")

    if uids is not None:
        if len(uids) != batch_size:
            raise ValueError("UID count does not match batch")

        if len(set(map(str, uids))) != batch_size:
            raise ValueError(
                "Duplicate UID in batch: "
                "use unique reports for this baseline."
            )

    # Similarity matrix: [B, B]
    logits = (
        image_embeddings @ text_embeddings.T
    ) / temperature

    # The correct pairs are on the diagonal
    labels = torch.arange(
        batch_size,
        device=logits.device
    )

    # Image -> Text
    loss_i2t = F.cross_entropy(
        logits, labels
    )

    # Text -> Image
    loss_t2i = F.cross_entropy(
        logits.T, labels
    )

    return (loss_i2t + loss_t2i) / 2
