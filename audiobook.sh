#!/bin/bash
# audiobook.sh — run one audiobook job detached and read it back in one line.
#
#   ./audiobook.sh start 1210-1261 --split 10,10,10,10,12   # or: --group 10 --merge-tail
#   ./audiobook.sh status        # one line: RUNNING n/N chapters, f/F files, ETA | DONE | FAILED | IDLE
#   ./audiobook.sh wait          # block until the job ends, print its result; exit 0 = DONE
#   ./audiobook.sh tail [N]      # last N log lines (default 20), for when status says FAILED
#   ./audiobook.sh oov 1462-1561     # words Kokoro guesses at: lexicon candidates (seconds)
#   ./audiobook.sh check 1262-1271   # Whisper round trip on rendered chapters (~15 s each, foreground)
#
# start finds the archived text file that holds the whole range, refuses while another job is
# running, and returns as soon as the job has printed its plan. Any further flags go to
# `audiobook_maker.py make` (--voice, --speed, --dry-run, ...). A macOS notification fires when
# the job ends, pass or fail. Env: SHADOWSLAVE_DIR (default ~/Desktop/ShadowSlave),
# AUDIOBOOK_STATE (default ~/.cache/audiobook-maker).
set -uo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
SS="${SHADOWSLAVE_DIR:-$HOME/Desktop/ShadowSlave}"
STATE="${AUDIOBOOK_STATE:-$HOME/.cache/audiobook-maker}"
PIDF="$STATE/job.pid" LOG="$STATE/job.log" EXITF="$STATE/job.exit" STARTF="$STATE/job.started"

die() { echo "audiobook: $*" >&2; exit 2; }

running() { [ -f "$PIDF" ] && [ ! -f "$EXITF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

fmt() {  # seconds -> 1h02m / 14m05s
    local s=$1
    if [ "$s" -ge 3600 ]; then printf '%dh%02dm' $((s / 3600)) $((s % 3600 / 60))
    else printf '%dm%02ds' $((s / 60)) $((s % 60)); fi
}

find_txt() {  # the archived file whose chapter span covers lo..hi
    local lo=$1 hi=$2 f a b
    for f in "$SS"/archived/*_chapters_*_*.txt; do
        [ -e "$f" ] || continue
        a=$(basename "$f" .txt | sed -E 's/.*_chapters_([0-9]+)_([0-9]+)$/\1/')
        b=$(basename "$f" .txt | sed -E 's/.*_chapters_([0-9]+)_([0-9]+)$/\2/')
        if [ "$lo" -ge "$a" ] && [ "$hi" -le "$b" ]; then echo "$f"; return 0; fi
    done
    return 1
}

status_line() {
    if [ ! -f "$LOG" ]; then echo "IDLE: no job has run"; return 0; fi
    local plan total new files done_ch done_new wrote now started elapsed eta rc
    plan=$(grep -m1 'plan:' "$LOG" | sed -E 's/^\[[0-9:]+\] //')
    total=$(sed -nE 's/.*plan: ([0-9]+) chapters.*/\1/p' <<<"$plan")
    new=$(sed -nE 's/.*\(([0-9]+) new.*/\1/p' <<<"$plan")
    files=$(sed -nE 's/.*, ([0-9]+) file\(s\), est.*/\1/p' <<<"$plan")
    done_ch=$(grep -cE '^\[[0-9:]+\] ch +[0-9]+ .*(realtime|cached)$' "$LOG")
    done_new=$(grep -cE 'realtime$' "$LOG")
    wrote=$(grep -c '\] wrote ' "$LOG")
    if [ -f "$EXITF" ]; then
        rc=$(cat "$EXITF")
        if [ "$rc" = 0 ]; then echo "DONE: $(grep 'RESULT:' "$LOG" | tail -1 | sed -E 's/^\[[0-9:]+\] //')"
        else echo "FAILED (exit $rc) after $done_ch/${total:-?} chapters: $(grep -vE '^\s*$' "$LOG" | tail -1)"; fi
        return 0
    fi
    if ! running; then echo "FAILED: job process is gone without writing an exit code; ./audiobook.sh tail"; return 0; fi
    now=$(date +%s); started=$(cat "$STARTF"); elapsed=$((now - started))
    if [ -z "$plan" ]; then echo "RUNNING: starting up ($(fmt $elapsed))"; return 0; fi
    if [ "$done_new" -gt 0 ] && [ -n "$new" ]; then
        eta=$(fmt $(( elapsed * (new - done_new) / done_new )))
    else
        eta="$(sed -nE 's/.*est ~(.*)$/\1/p' <<<"$plan") (plan)"
    fi
    echo "RUNNING: $done_ch/$total chapters, $wrote/$files files, elapsed $(fmt $elapsed), ETA ~$eta"
}

cmd="${1:-status}"; shift || true
case "$cmd" in
start)
    [ $# -ge 1 ] || die "usage: start N-M [make flags...]"
    range=$1; shift
    [[ "$range" =~ ^([0-9]+)-([0-9]+)$ ]] || die "range must look like 1210-1261, got '$range'"
    lo=${BASH_REMATCH[1]} hi=${BASH_REMATCH[2]}
    running && die "a job is already running (pid $(cat "$PIDF")): $(status_line)"
    txt=$(find_txt "$lo" "$hi") || die "no file in $SS/archived covers $lo-$hi (ranges cannot span two files)"
    mkdir -p "$STATE"; rm -f "$EXITF"; : > "$LOG"; date +%s > "$STARTF"
    # The job shell: run make, record its exit code, notify. Detached with a new session so the
    # caller's shell (or a Claude tool call) ending cannot take it down; macOS has no setsid.
    job='cd "$1" && uv run audiobook_maker.py make "${@:5}" > "$2" 2>&1; rc=$?; echo $rc > "$3"
         msg=$(grep "RESULT:" "$2" | tail -1 | sed -E "s/^\[[0-9:]+\] //"); [ $rc = 0 ] || msg="FAILED (exit $rc)"
         osascript -e "display notification \"$4: $msg\" with title \"Audiobook\"" >/dev/null 2>&1 || true'
    python3 - "$PIDF" /bin/bash -c "$job" job "$REPO" "$LOG" "$EXITF" "Ch $lo-$hi" \
        "$txt" --out-dir "$SS/audiobooks" --chapters "$lo-$hi" "$@" <<'EOF'
import subprocess, sys
p = subprocess.Popen(sys.argv[2:], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
open(sys.argv[1], "w").write(f"{p.pid}\n")
EOF
    for _ in $(seq 120); do  # up to 60 s for the plan line (or an early failure)
        grep -q 'plan:' "$LOG" 2>/dev/null && break
        [ -f "$EXITF" ] && break
        sleep 0.5
    done
    echo "$(status_line) | pid $(cat "$PIDF") | log $LOG"
    ;;
status) status_line ;;
wait)
    while running; do sleep 15; done
    line=$(status_line); echo "$line"
    [[ "$line" == DONE* ]]
    ;;
tail) tail -n "${1:-20}" "$LOG" 2>/dev/null || echo "IDLE: no job has run" ;;
oov|check)
    [ $# -ge 1 ] || die "usage: $cmd N-M [flags...]"
    range=$1; shift
    [[ "$range" =~ ^([0-9]+)-([0-9]+)$ ]] || die "range must look like 1262-1271, got '$range'"
    txt=$(find_txt "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}") || die "no file in $SS/archived covers $range"
    [ "$cmd" = check ] && running && die "a render job is running; check shares the GPU with it — wait for it first"
    extra=(); [ "$cmd" = check ] && extra=(--out-dir "$SS/audiobooks")
    cd "$REPO" && exec uv run audiobook_maker.py "$cmd" "$txt" --chapters "$range" ${extra[@]+"${extra[@]}"} "$@"
    ;;
record)
    cd "$REPO" && exec uv run python record.py "$@"
    ;;
*) die "unknown command '$cmd' (start | status | wait | tail | oov | check | record)" ;;
esac
