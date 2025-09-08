# Mapping between index and letter (A–Z)
idx_to_letter = {i: chr(ord('A') + i) for i in range(26)}
letter_to_idx = {v: k for k, v in idx_to_letter.items()}
