from src.models.model_ensemble import EnsembleClassifier
from src.utils.window import compute_llps, compute_llps_window, compute_llps_mean


def get_delta_llps(
    model, 
    ref_sequence: str, 
    ref_saprot_sequence: str, 
    alt_sequence: str,
    alt_saprot_sequence: str,
    pos_aa: int,
    mode: str = "protein_level"
):
    if not isinstance(model, EnsembleClassifier):
        raise NotImplementedError(f"Only support ensemble model for mutagenesis analysis.")
    if mode == "protein_level":
        scale = 100
        ref_prob = compute_llps(model, ref_sequence, ref_saprot_sequence) * scale
        alt_prob = compute_llps(model, alt_sequence, alt_saprot_sequence) * scale
    elif mode == "region_level_fast":
        ref_prob = compute_llps_window(model, ref_sequence, ref_saprot_sequence, pos_aa-1)
        alt_prob = compute_llps_window(model, alt_sequence, alt_saprot_sequence, pos_aa-1)
    else: # mode == "region_level"
        scale = len(ref_sequence)
        ref_prob = compute_llps_mean(model, ref_sequence, ref_saprot_sequence) * scale
        alt_prob = compute_llps_mean(model, alt_sequence, alt_saprot_sequence) * scale
    # mutation score = LLPS(mutation) - LLPS(wild type)
    return alt_prob - ref_prob


if __name__ == '__main__':
    # LLPS models
    import torch
    from src.lib.llps_model import get_llps_model
    # input
    from src.utils.saprotseq import get_saprotseq

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model
    model = get_llps_model('ensemble', local_files_only=True)
    model = model.to(device)
    # input
    # sequence: str
    # saprotseq: str
    ref_sequence = 'MEELSADEIRRRRLARLAGGQTS'
    alt_sequence = 'MEELSDDEIRRRRLARLAGGQTS'
    ref_saprot_sequence = get_saprotseq(ref_sequence)
    alt_saprot_sequence = get_saprotseq(alt_sequence)
    pos_aa = [i for i, (ref, alt) in enumerate(zip(ref_sequence, alt_sequence)) if ref != alt][0] + 1
    # inference
    score = get_delta_llps(model, ref_sequence, ref_saprot_sequence, alt_sequence, alt_saprot_sequence, pos_aa)
    print("Sequence length:", len(ref_sequence))
    print("Wild type sequence:", ref_sequence)
    print("Mutant sequence:", alt_sequence)
    print("Mutation position:", pos_aa)
    print("Delta LLPS score:", score)
