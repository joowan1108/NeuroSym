# Vast.ai template "On-start Script": clone the code, then fetch assets and checkpoints
# in the background (progress: tail -f /workspace/download.log).
cd /workspace
if [ ! -d NeuroSym ]; then
  git clone "https://${GITHUB_TOKEN:+${GITHUB_TOKEN}@}github.com/joowan1108/NeuroSym.git"
fi
cd NeuroSym && git pull --ff-only || true
nohup bash setup/download.sh > /workspace/download.log 2>&1 &
