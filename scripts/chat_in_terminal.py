import argparse
import sys
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive chat with a local Hugging Face model."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name or local path.",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Optional system prompt to prepend to the chat.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per turn.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling cutoff.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling cutoff.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device placement mode.",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Enable 4-bit quantization (requires GPU).",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable tokenizer chat templates and use a plain prompt.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading remote code from the model repo.",
    )
    return parser.parse_args()


def build_prompt(
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    use_chat_template: bool,
) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    lines = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "system":
            lines.append(f"System: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"User: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)


def resolve_quantization(use_4bit: bool) -> object | None:
    if not use_4bit:
        return None
    if not torch.cuda.is_available():
        raise SystemExit("4-bit quantization requires CUDA. Use --device cpu.")
    from transformers import BitsAndBytesConfig

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def resolve_device_map(device: str) -> tuple[str, object]:
    if device == "cpu":
        return "cpu", torch.float32
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Use --device cpu.")
    return "auto", "auto"


def print_help() -> None:
    print("Commands:")
    print("  /help  Show this message")
    print("  /reset Clear chat history")
    print("  /exit  Quit the chat")
    print("  /quit  Quit the chat")


def main() -> int:
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device_map, torch_dtype = resolve_device_map(args.device)
    quantization_config = resolve_quantization(args.use_4bit)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=device_map,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    use_chat_template = not args.no_chat_template
    messages: List[Dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    print("Interactive chat ready. Type /help for commands.")

    while True:
        try:
            user_text = input("User> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            break
        if user_text == "/help":
            print_help()
            continue
        if user_text == "/reset":
            messages = []
            if args.system:
                messages.append({"role": "system", "content": args.system})
            print("Chat history cleared.")
            continue

        messages.append({"role": "user", "content": user_text})
        prompt = build_prompt(tokenizer, messages, use_chat_template)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        do_sample = args.temperature > 0
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
                top_p=args.top_p if do_sample else None,
                top_k=args.top_k if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        print(f"Assistant> {reply}")
        messages.append({"role": "assistant", "content": reply})

    return 0


if __name__ == "__main__":
    sys.exit(main())
