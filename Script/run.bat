
@echo off
set folder=%1
cd D:\gpt2\TensorRT-LLM\examples\gpt

python ..\run.py --input_text "It is the humane way: the other course Will prove too bloody, and the end of it Unknown to the beginning." --engine_dir=.\data\%folder%\engine_output --max_output_len 128
