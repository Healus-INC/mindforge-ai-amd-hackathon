# Qwen / Unsloth Fine-Tuning Notes

## Format

The training files use the common Hugging Face conversational SFT format:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "{...valid JSON...}"}
  ]
}
```

## Recommended base models for fast hackathon path

Use one of:

- Qwen/Qwen2.5-1.5B-Instruct for fastest test
- Qwen/Qwen2.5-3B-Instruct if GPU and time allow
- Qwen/Qwen2.5-7B-Instruct only if Harris's AMD setup is stable

## Training objective

Supervised fine-tuning to enforce:

1. JSON-only output
2. Conservative risk classification
3. consistent escalation language
4. patient-safe and clinician-ready summaries
5. adherence + side-effect + sleep/mood extraction

## Validation targets

- Valid JSON rate
- Risk-level accuracy on held-out cases
- Correct escalation level for crisis/high-risk scenarios
- No diagnosis/prescribing language
- Mentions emergency/crisis escalation when required

## Fast fallback

If LoRA fine-tuning becomes slow, use the same data for:

- few-shot prompting
- unit tests
- eval table in pitch deck
- demo cases in Gradio
