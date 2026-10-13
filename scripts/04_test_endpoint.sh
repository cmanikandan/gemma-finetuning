#!/usr/bin/env bash
# ==============================================================================
# LAB 7: Smoke-Test Your Deployed Sovereign FSI Gemma Endpoint
# ==============================================================================
# Sends 3 real Indian FSI regulatory questions to your live vLLM endpoint and
# prints the model's answers, so you can visually confirm the fine-tuning worked.
#
# Run this AFTER ./scripts/03_deploy_sovereign_vllm.sh completes.
# ==============================================================================

set -uo pipefail

export PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
export REGION="${REGION:-asia-south1}"
export ENDPOINT_DISPLAY_NAME="${ENDPOINT_DISPLAY_NAME:-fsi-sovereign-vllm-endpoint}"

echo "================================================================================"
echo " SMOKE TEST — SOVEREIGN FSI GEMMA ENDPOINT (${REGION})"
echo "================================================================================"

# ------------------------------------------------------------------------------
# Locate the endpoint
# ------------------------------------------------------------------------------
ENDPOINT_ID=$(gcloud ai endpoints list --region="${REGION}" --project="${PROJECT_ID}" \
  --filter="displayName:${ENDPOINT_DISPLAY_NAME}" --format="value(name)" 2>/dev/null | head -n 1)

if [[ -z "${ENDPOINT_ID}" ]]; then
  echo "[!] ERROR: No endpoint named '${ENDPOINT_DISPLAY_NAME}' found in ${REGION}."
  echo "    Deploy one first: ./scripts/03_deploy_sovereign_vllm.sh"
  exit 1
fi
echo "[+] Found endpoint: ${ENDPOINT_ID}"

# Confirm a model is actually deployed and serving
DEPLOYED=$(gcloud ai endpoints describe "${ENDPOINT_ID}" --region="${REGION}" \
  --project="${PROJECT_ID}" --format="value(deployedModels[].displayName)" 2>/dev/null)
if [[ -z "${DEPLOYED}" ]]; then
  echo "[!] ERROR: Endpoint exists but has no deployed model."
  echo "    Deployment can take 15-25 minutes. Check the Console, then retry."
  exit 1
fi
echo "[+] Deployed model: ${DEPLOYED}"

ENDPOINT_URL="https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/${ENDPOINT_ID}:predict"

# ------------------------------------------------------------------------------
# Three representative Indian FSI test prompts, one per regulatory domain
# ------------------------------------------------------------------------------
SYSTEM_PROMPT="You are a Principal Financial Regulatory & Operations AI Assistant specialized in Indian Banking, Financial Services, and Insurance (BFSI) compliance. Your responses must strictly adhere to Reserve Bank of India (RBI), Securities and Exchange Board of India (SEBI), and National Payments Corporation of India (NPCI) master circulars."

send_prompt() {
  local test_name="$1"
  local user_prompt="$2"
  local expected_signal="$3"

  echo ""
  echo "--------------------------------------------------------------------------------"
  echo " TEST: ${test_name}"
  echo " Looking for: ${expected_signal}"
  echo "--------------------------------------------------------------------------------"

  # Build the Gemma 4 chat-template prompt (<|turn>system...<turn|><|turn>user...<turn|><|turn>model)
  local full_prompt="<bos><|turn>system
${SYSTEM_PROMPT}<turn|>
<|turn>user
${user_prompt}<turn|>
<|turn>model
"

  # Write the request body with python3 so quotes/newlines are escaped correctly
  python3 -c "
import json, sys
body = {
  'instances': [{
    'prompt': sys.argv[1],
    'messages': [
      {'role': 'system', 'content': sys.argv[2]},
      {'role': 'user', 'content': sys.argv[3]}
    ],
    'max_tokens': 512,
    'temperature': 0.1,
    'top_p': 0.95
  }]
}
print(json.dumps(body))
" "${full_prompt}" "${SYSTEM_PROMPT}" "${user_prompt}" > /tmp/fsi_request.json

  START_MS=$(python3 -c "import time; print(int(time.time()*1000))")
  RESPONSE=$(curl -s -X POST \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -H "Content-Type: application/json" \
    "${ENDPOINT_URL}" \
    -d @/tmp/fsi_request.json)
  END_MS=$(python3 -c "import time; print(int(time.time()*1000))")

  echo "  Latency: $((END_MS - START_MS)) ms"
  echo ""
  echo "  MODEL RESPONSE:"
  echo "${RESPONSE}" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    preds = data.get('predictions', [])
    if not preds:
        print('  [!] No predictions returned. Raw response:')
        print('  ' + json.dumps(data)[:800])
    else:
        text = preds[0] if isinstance(preds[0], str) else json.dumps(preds[0])
        for line in text.split('\n'):
            print('  | ' + line)
except Exception as e:
    print('  [!] Could not parse response:', e)
    print('  ' + sys.stdin.read()[:800] if not sys.stdin.closed else '')
"
}

send_prompt \
  "1. NPCI UPI Dispute — RBI Turnaround Time Compensation" \
  "A UPI P2M transaction of Rs 12,500 (UTR 426010000123) was debited but the merchant was never credited. It is now 9 calendar days since the transaction date and it remains unresolved. Calculate the exact statutory compensation owed and cite the governing RBI circular." \
  "Should cite RBI/2019-20/67 and calculate Rs 100/day penalty for days beyond T+5"

send_prompt \
  "2. RBI KYC/AML — Ultimate Beneficial Owner Threshold" \
  "For a Private Limited Company customer classified as High Risk, what is the statutory Ultimate Beneficial Owner (UBO) identification threshold, the mandatory Re-KYC cycle, and the FIU-IND STR filing timeline?" \
  "Should state 10% UBO threshold, 2-year Re-KYC for high risk, and 7-day STR filing under Rule 8(2)"

send_prompt \
  "3. Digital Lending — Key Fact Statement APR Computation" \
  "A digital personal loan of Rs 1,00,000 for 12 months at 18% nominal interest has a 2.5% upfront processing fee plus 18% GST. The LSP proposes a 12% FLDG and disbursement via its nodal escrow account. Compute the KFS APR and audit the structure against RBI Digital Lending Guidelines." \
  "Should flag the 12% FLDG as exceeding the 5% cap and reject the LSP nodal escrow routing"

echo ""
echo "================================================================================"
echo " SMOKE TEST COMPLETE"
echo ""
echo " WHAT TO LOOK FOR:"
echo "   A well fine-tuned model cites EXACT circular numbers (RBI/2019-20/67),"
echo "   computes EXACT rupee amounts, and produces structured runbooks."
echo "   A base/untuned model gives vague generic banking advice with no citations."
echo ""
echo " >>> COST REMINDER <<<"
echo " This endpoint bills ~\$0.85/hour while it exists, even when idle."
echo " When you are done experimenting, run:"
echo "     ./scripts/99_cleanup_resources.sh --confirm"
echo "================================================================================"
