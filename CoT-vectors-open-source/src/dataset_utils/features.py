"""Pure-Python token construction; tensor conversion stays in the data loader."""


def left_pad(sequences, pad_value):
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("Cannot pad an empty batch or sequence")
    width = max(map(len, sequences))
    return [[pad_value] * (width - len(sequence)) + list(sequence) for sequence in sequences]


def training_features(examples, tokenizer, max_seq_len):
    """Keep answer IDs identical in teacher/student inputs, including at BPE boundaries.

    Chat templates already contain their special tokens. Tokenize the answer as an
    explicit segment in both inputs rather than searching unrelated question tokens
    or concatenating non-contiguous LCS matches. Left truncation preserves the answer.
    """
    if max_seq_len < 2:
        raise ValueError("max_seq_len must be at least two")
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define padding and EOS tokens")

    def encode(text):
        return list(tokenizer(text, add_special_tokens=False).input_ids)

    teacher_ids, teacher_masks, cot_masks = [], [], []
    student_ids, student_labels = [], []
    for example in examples:
        response, answer = example["response"], str(example["answer"])
        if not answer or example["teacher_text"].rfind(response) < 0:
            raise ValueError("Teacher response or final answer is missing")
        local_start = response.rfind(answer)
        if local_start < 0:
            raise ValueError("The final answer is absent from the teacher response")
        response_start = example["teacher_text"].rfind(response)
        answer_start = response_start + local_start
        answer_end = answer_start + len(answer)
        answer_ids = encode(answer)
        if not answer_ids:
            raise ValueError("The final answer tokenizes to an empty sequence")
        head = encode(example["teacher_text"][:response_start])
        cot = encode(example["teacher_text"][response_start:answer_start])
        tail = encode(example["teacher_text"][answer_end:])
        if len(answer_ids) + len(tail) >= max_seq_len:
            raise ValueError("The teacher answer exceeds max_seq_len; increase the limit")
        prefix = head + cot
        full_teacher = prefix + answer_ids + tail
        teacher_mask = [False] * len(prefix) + [True] * len(answer_ids) + [False] * len(tail)
        cot_mask = [False] * len(head) + [True] * (len(cot) + len(answer_ids)) + [False] * len(tail)
        teacher_ids.append(full_teacher[-max_seq_len:])
        teacher_masks.append(teacher_mask[-max_seq_len:])
        cot_masks.append(cot_mask[-max_seq_len:])

        prompt_ids = encode(example["student_text"])
        if len(answer_ids) + 1 >= max_seq_len or not prompt_ids:
            raise ValueError("The student answer exceeds max_seq_len; increase the limit")
        full_student = prompt_ids + answer_ids + [tokenizer.eos_token_id]
        labels = [-100] * len(prompt_ids) + answer_ids + [-100]
        student_ids.append(full_student[-max_seq_len:])
        student_labels.append(labels[-max_seq_len:])

    teacher_attention = left_pad([[1] * len(ids) for ids in teacher_ids], 0)
    student_attention = left_pad([[1] * len(ids) for ids in student_ids], 0)
    return {
        "prefix_inputs": {"input_ids": left_pad(teacher_ids, tokenizer.pad_token_id),
                          "attention_mask": teacher_attention},
        "teacher_answer_mask": left_pad(teacher_masks, False),
        "teacher_cot_mask": left_pad(cot_masks, False),
        "student_inputs": {"input_ids": left_pad(student_ids, tokenizer.pad_token_id),
                           "attention_mask": student_attention},
        "student_labels": left_pad(student_labels, -100),
        "images": [[] for _ in examples],
    }
