# Code Repository of Workshop Submission: _ALPS-Lab at SemEval‑2026 Task 3_

## Task 3 reproduce

Most code store on `/starter-kit/task2task3/LLM-Based method`.

`python mkzip.py <directory>`: archive all files and directories under \<directory>.

To produce the predictions:

```bash
pip install unsloth
for i in restaurant laptop hotel
do
  python mian.py train --task 3 $i
done
python mian.py infer --task 3 restaurant rus
python mian.py infer --task 3 restaurant tat
python mian.py infer --task 3 restaurant ukr
python mian.py infer --task 3 restaurant zho
python mian.py infer --task 3 restaurant eng
python mian.py infer --task 3 laptop zho
python mian.py infer --task 3 laptop eng
python mian.py infer --task 3 hotel jpn
```
