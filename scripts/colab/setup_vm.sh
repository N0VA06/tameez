#!/usr/bin/env bash
# Install the exact evaluation stack used for every run (versions recorded in each results_*.json).
pip install -q "vllm==0.33.0" "transformers==5.20.2" "lm_eval[ifeval,math]==0.4.13" langdetect immutabledict
python -c "import vllm, transformers, torch, lm_eval; print('VERSIONS', vllm.__version__, transformers.__version__, torch.__version__)"
pip list 2>/dev/null | grep -iE "^(lm.eval|lm_eval) "
echo SETUP_DONE
