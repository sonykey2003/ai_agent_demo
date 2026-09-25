#!/usr/bin/env bash
#
# Drive chat traffic at the backend so traces, logs and DB activity appear in the
# configured backends. Bank is excluded by default: that domain runs the native
# Galileo path with OTel muted, so it emits no spans.
#
#   ./scripts/loadgen.sh                 # 5 rounds over the default domains
#   ITERATIONS=20 ./scripts/loadgen.sh
#   DOMAINS="healthcare" ./scripts/loadgen.sh
#   BASE_URL=http://localhost:8800 DELAY=5 ./scripts/loadgen.sh

base_url="${BASE_URL:-http://localhost:8800}"
domains="${DOMAINS:-healthcare insurance restaurant platform}"
iterations="${ITERATIONS:-5}"
delay="${DELAY:-2}"

prompts_for() {
    case "$1" in
        healthcare)
            echo "What is covered for water damage under my plan?
How do I request a referral to a specialist?
What happens if I miss a premium payment?" ;;
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
        *)
            echo "How is this demo architected?
Which observability backends are wired up?
How does retrieval-augmented generation work here?" ;;
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
