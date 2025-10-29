conda create -n dllm-var python=3.12
pip install -r requirements.txt 

# Inference dLLM-Var model from Hugging Face
huggingface-cli downloads maomaocun/dLLM-Var --local-dir \path\to\dLLM-Var
python demo_dLLM_Var.py --model_name_or_path \path\to\dLLM-Var --input_text "Your input text here"
