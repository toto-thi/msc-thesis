from typing import Dict, Any, List
from tqdm import tqdm
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage

import json, re, time, base64

def add_trace(state: dict, *, agent: str, role: str, payload):
    if "trace" not in state:
        state["trace"] = []
    state["trace"].append({
        "agent": agent,
        "role": role,
        "payload": payload,
    })

def extract_first_json(text: Any) -> Dict[str, Any]:
    """
    Extract the first JSON object from `text`.
    - Handles ```json fences.
    - Finds the first balanced {...} with brace counting.
    - Tries lightweight repairs (auto-closing braces).
    - Always returns a dict: parsed JSON or {"_raw": ..., "_reason": ...}.
    """
    # Fast paths for non-strings
    if isinstance(text, dict):
        return text
    if text is None:
        return {"_raw": "", "_reason": "none_input"}
    if not isinstance(text, str):
        return {"_raw": str(text), "_reason": f"not_a_string:{type(text).__name__}"}

    original = text
    s = text.strip()
    
    # Strip leading ``` or ```json fences and a trailing closing fence
    s = re.sub(r"^```[\w-]*\s*\n", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\n```[\s\t]*$", "", s)

    # 0) Whole-string JSON
    try:
        val = json.loads(s)
        return val if isinstance(val, dict) else {"_raw": s, "_reason": "json_root_not_object"}
    except Exception:
        pass

    # 1) Find first balanced { ... }
    start = s.find("{")
    if start == -1:
        return {"_raw": s, "_reason": "no_brace_found"}

    depth = 0
    in_string = False
    escape = False
    end = None

    for i, ch in enumerate(s[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

    candidate = s[start:(end + 1) if end is not None else len(s)]

    # 2) Parse balanced candidate
    if end is not None:
        try:
            val = json.loads(candidate)
            return val if isinstance(val, dict) else {"_raw": candidate, "_reason": "json_root_not_object"}
        except Exception as e:
            return {"_raw": candidate, "_reason": f"parse_error_balanced:{type(e).__name__}"}

    # 3) Try auto-closing braces (only if not inside a string)
    if not in_string and depth > 0:
        repaired = candidate + ("}" * depth)
        try:
            val = json.loads(repaired)
            return val if isinstance(val, dict) else {"_raw": repaired, "_reason": "json_root_not_object"}
        except Exception as e:
            return {"_raw": repaired, "_reason": f"parse_error_repaired:{type(e).__name__}"}

    # 4) Last resort
    return {"_raw": original, "_reason": "unbalanced_in_string_or_unknown"}

def encode_image(image_path):
    """Encode a local image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def invoke_llm(model, SYSTEM_PROMPT, user_message):

    messages =[
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message)
    ]

    response = model.invoke(messages)
    return extract_first_json(response.text())

def run_evaluation_in_batch(
    app,
    test_dataset: List[Dict[str, Any]],
    out_dir: str,
    out_file: str,
    *,
    resume: bool = True,
    log_interval: int = 10,
) -> Path:
    """
    Runs the agent over a list of items shaped like:
        {"img_path": "...", "diagnosis": "...", "melanocytic": Optional[bool]}
    and writes the *final agent state* for each item as one JSON line to:
        <out_dir>/agent_evaluation_full.jsonl

    - No timeout.
    - Resume-safe: skips items already present in the JSONL (by image stem).
    - Returns the Path to the JSONL file.
    """
    print("🚀 Starting evaluation (JSONL only)…")
    print(f"📊 Dataset size: {len(test_dataset)}")
    print(f"🔄 Resuming: {resume}")

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir_p / out_file

    processed_ids = set()
    if resume and jsonl_path.exists():
        print(f"🔁 Resuming from {jsonl_path}")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    st = json.loads(line)
                    # Accept either 'image_path' or 'image_path' for robustness
                    p = st.get("image_path") or st.get("image_path") or ""
                    if p:
                        processed_ids.add(Path(p).stem)
                except Exception:
                    # ignore malformed lines during resume scan
                    pass
        print(f"✅ Already processed: {len(processed_ids)} images")

    # Filter items to process
    unprocessed = [it for it in test_dataset if Path(it["image_path"]).stem not in processed_ids]
    total = len(test_dataset)
    remaining = len(unprocessed)
    print(f"⏭️  To process: {remaining}/{total}")

    # Open JSONL file for append or write fresh
    mode = "a" if resume and jsonl_path.exists() else "w"
    n_ok, n_fail = 0, 0

    with open(jsonl_path, mode, encoding="utf-8") as jf:
        if remaining > 0:
            iterator = tqdm(unprocessed, desc="Processing Images", unit="img")
            start_time = time.time()

            for i, item in enumerate(iterator):
                image_stem = Path(item["image_path"]).stem
                try:
                    
                    state = {
                        "image_path": item["image_path"],
                        "patient_data": {
                            "age": item.get("age", ""),
                            "sex": item.get("sex", ""),
                            "lesion_location": item.get("anatom_site", ""),
                            "is_melanocytic": item.get("melanocytic", ""),
                        },
                        "ground_truth": item.get("diagnosis") or item.get("label") or "",
                        "trace": [],
                    }
                    
                    run_id = int(time.time())
                    config = {"configurable": {"thread_id": f"case-{image_stem}-{run_id}"}}
                    
                    final_state = app.invoke(state, config=config)

                    # Ensure resume keys are present in the saved line
                    if "image_path" not in final_state:
                        final_state["image_path"] = item["image_path"]
                    if "ground_truth" not in final_state:
                        final_state["ground_truth"] = state["ground_truth"]

                    jf.write(json.dumps(final_state, ensure_ascii=False) + "\n")
                    jf.flush()
                    n_ok += 1

                    # Progress / ETA
                    if (i + 1) % log_interval == 0:
                        elapsed = time.time() - start_time
                        avg = elapsed / (i + 1)
                        eta = avg * (remaining - i - 1)
                        print(f"⏳ Progress: {i+1}/{remaining} | ETA: {int(eta//60)}m {int(eta%60)}s")

                except Exception as e:
                    err = f"{type(e).__name__}: {str(e)}"
                    print(f"❌ Failed on {image_stem}: {err}")
                    patient_data = {
                        "age": item.get("age", ""),
                        "sex": item.get("sex", ""),
                        "lesion_location": item.get("anatom_site", ""),
                    }
                    error_state = {
                        "image_path": item["image_path"],
                        "patient_data": patient_data,
                        "ground_truth": item.get("diagnosis") or item.get("label") or "",
                        "melanocytic_gt": item.get("melanocytic", ""),
                        "_error": err,
                        "trace": [],
                    }
                    jf.write(json.dumps(error_state, ensure_ascii=False) + "\n")
                    jf.flush()
                    n_fail += 1

    # Summary
    print("\n" + "="*48)
    print("✅ EVALUATION COMPLETE (JSONL only)")
    print(f"   Total items: {total}")
    print(f"   Newly processed: {n_ok}")
    print(f"   Newly failed:    {n_fail}")
    print(f"   JSONL saved to:  {jsonl_path}")
    print("="*48)

    return jsonl_path