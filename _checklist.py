import os

checks = {}

# 1. Required files
for f in ['inference.py','environment.py','tasks.py','app.py','openenv.yaml','Dockerfile','requirements.txt']:
    checks[f] = os.path.exists(f)

# 2. Env vars / OpenAI client / log format
src = open('inference.py', encoding='utf-8').read()
checks['API_BASE_URL used']   = 'API_BASE_URL' in src
checks['MODEL_NAME used']     = 'MODEL_NAME' in src
checks['HF_TOKEN used']       = 'HF_TOKEN' in src
checks['OpenAI client']       = 'from openai import OpenAI' in src
checks['[START] emitted']     = '[START]' in src
checks['[STEP] emitted']      = '[STEP]' in src
checks['[END] emitted']       = '[END]' in src

# 3. Static dashboard
checks['static/index.html'] = os.path.exists(os.path.join('static','index.html'))
checks['static/style.css']  = os.path.exists(os.path.join('static','style.css'))

# 4. tasks.py structure
tasks_src = open('tasks.py', encoding='utf-8').read()
checks['TASK_REGISTRY']   = 'TASK_REGISTRY' in tasks_src
checks['3 tasks defined'] = tasks_src.count('task_id=') >= 3

# 5. openenv.yaml has docker section and 3 task IDs
yaml_src = open('openenv.yaml', encoding='utf-8').read()
checks['yaml: docker section'] = 'docker:' in yaml_src
checks['yaml: 3 task entries']  = yaml_src.count('- id: task_') >= 3
checks['yaml: /reset endpoint'] = '/reset' in yaml_src
checks['yaml: /step endpoint']  = '/step' in yaml_src
checks['yaml: /state endpoint'] = '/state' in yaml_src

# 6. Dockerfile
df = open('Dockerfile', encoding='utf-8').read()
checks['Dockerfile: EXPOSE 7860'] = 'EXPOSE 7860' in df
checks['Dockerfile: uvicorn CMD']  = 'uvicorn' in df
checks['Dockerfile: static COPY']  = 'static/' in df

print()
for k, v in checks.items():
    print(f"  {'OK  ' if v else 'FAIL'}  {k}")

fails = sum(1 for v in checks.values() if not v)
print()
print('='*50)
print(f"{'ALL PASS' if fails == 0 else str(fails) + ' FAILURE(S)'}")
print('='*50)
