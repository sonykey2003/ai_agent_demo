#!/usr/bin/env bash
#
# Drive chat traffic at the backend so traces, logs and DB activity appear in the
# configured backends. Bank, ecomm, and healthcare are excluded by default: those
# domains run the native Galileo path (Agent Control native domains), so their
# OTel spans go to Splunk APM only. Include them with DOMAINS="bank ecomm healthcare".
#
#   ./scripts/loadgen.sh                 # 5 rounds over the default domains
#   ITERATIONS=20 ./scripts/loadgen.sh
#   DOMAINS="insurance" ./scripts/loadgen.sh
#   BASE_URL=http://localhost:8800 DELAY=5 ./scripts/loadgen.sh

base_url="${BASE_URL:-http://localhost:8800}"
domains="${DOMAINS:-insurance restaurant}"
iterations="${ITERATIONS:-5}"
delay="${DELAY:-2}"

prompts_for() {
    case "$1" in
        healthcare)
            echo "What are the common side effects of Metformin?
Which prescriptions are due for a refill?
What monitoring is required when taking Atorvastatin?" ;;
        insurance)
            echo "How do I file a claim after a car accident?
What is my deductible for storm damage?
Does my policy cover a rental car while mine is repaired?" ;;
        restaurant)
            echo "What vegetarian options are on the menu?
Do you take reservations for large groups?
What are your opening hours on public holidays?" ;;
        bank)
            echo "How do I dispute a card transaction?
What is the daily transfer limit on my account?
How long does an international wire take?" ;;
        ecomm)
            echo "Can I return a bookcase I have already assembled?
Is the 3-seat fabric sofa in stock, and in which warehouse?
I am missing a screw from my dresser, what should I do?" ;;
        *)
            echo "What can you help me with?
How do I contact customer support?
What information do you need from me to get started?" ;;
    esac
}

if ! curl --fail --silent --max-time 5 "${base_url}/healthz" >/dev/null; then
    echo >&2 "Backend not reachable at ${base_url} — is 'docker compose up' running?"
    exit 1
fi

echo "Load generating against ${base_url} | domains: ${domains} | rounds: ${iterations}"

sent=0
failed=0
for round in $(seq 1 "$iterations"); do
    for domain in $domains; do
        # One prompt per round, cycling through that domain's list.
        prompt="$(prompts_for "$domain" | sed -n "$(( (round - 1) % 3 + 1 ))p")"
        payload=$(printf '{"message":%s,"domain":"%s"}' \
            "$(printf '%s' "$prompt" | sed 's/\\/\\\\/g; s/"/\\"/g; s/^/"/; s/$/"/')" \
            "$domain")

        start=$(date +%s)
        code=$(curl --silent --output /dev/null --max-time 180 \
            --write-out '%{http_code}' \
            -X POST "${base_url}/chat" \
            -H 'Content-Type: application/json' \
            -d "$payload")
        elapsed=$(( $(date +%s) - start ))

        sent=$(( sent + 1 ))
        if [ "$code" = "200" ]; then
            printf 'round %-3s %-12s %ss  %s\n' "$round" "$domain" "$elapsed" "$prompt"
        else
            failed=$(( failed + 1 ))
            printf 'round %-3s %-12s HTTP %s  %s\n' "$round" "$domain" "$code" "$prompt"
        fi

        sleep "$delay"
    done
done

echo "Done: ${sent} requests, ${failed} failed."
