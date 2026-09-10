from __future__ import annotations

BASH_INTEGRATION = r"""# Hunch Bash integration. Inspect this output, then: eval "$(hunch shell-init)"

_hunch_ms() {
  local t=${1:-0.0}
  local sec=${t%.*}
  local frac=${t#*.}000000
  printf '%s\n' $((10#$sec * 1000 + 10#${frac:0:6} / 1000))
}

_hunch_predict() {
  local start end output
  start=$(_hunch_ms "${EPOCHREALTIME:-0}")
  output=$(hunch predict 2>/dev/null) || true
  end=$(_hunch_ms "${EPOCHREALTIME:-0}")
  if ((end - start > 200)); then
    _HUNCH_AUTO=0
  fi
  output=${output%%$'\n'*}
  _HUNCH_SUGGESTION="$output"
}

_hunch_prompt() {
  if [[ ${_HUNCH_AUTO} -eq 0 ]]; then
    _HUNCH_SUGGESTION=
    return 0
  fi
  _hunch_predict
  if [[ -n "${_HUNCH_SUGGESTION}" ]]; then
    printf '%s\n' "${_HUNCH_SUGGESTION}"
    hunch record displayed >/dev/null 2>&1 || true
  fi
  return 0
}

_hunch_insert() {
  if [[ -n "${READLINE_LINE}" ]]; then
    return 0
  fi
  if [[ -z "${_HUNCH_SUGGESTION}" ]]; then
    _hunch_predict
  fi
  if [[ -z "${_HUNCH_SUGGESTION}" ]]; then
    return 0
  fi
  READLINE_LINE="${_HUNCH_SUGGESTION}"
  READLINE_POINT=${#READLINE_LINE}
  hunch record inserted >/dev/null 2>&1 || true
  return 0
}

if [[ -z ${_HUNCH_LOADED:-} ]]; then
  _HUNCH_LOADED=1
  _HUNCH_AUTO=1
  _HUNCH_SUGGESTION=
  if [[ ${PROMPT_COMMAND@a} == *a* ]]; then
    PROMPT_COMMAND+=(_hunch_prompt)
  else
    PROMPT_COMMAND="${PROMPT_COMMAND:+$PROMPT_COMMAND; }_hunch_prompt"
  fi
  bind -x '"\C-x\C-p": _hunch_insert' || true
fi
"""
