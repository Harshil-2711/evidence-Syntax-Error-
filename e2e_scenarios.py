import urllib.request, json

BASE = 'http://localhost:8000'
GOAL = 'Create project Hackathon Alpha, add Harshit as a member, make the project private, and generate a report.'

def call(method, path, body=None):
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(BASE+path, data=data, method=method, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

print('=== SCENARIO 1: Happy Path ===')
run = call('POST', '/workflow/run', {'goal': GOAL, 'max_retries': 3})
wid = run['workflow_id']
ev  = call('GET', '/workflow/' + wid + '/evidence')
au  = call('GET', '/workflow/' + wid + '/audit')
print('  status=' + run['final_status'] + ' verified=' + str(run['verified_actions']) + '/' + str(run['total_actions']) + ' contradictions=' + str(ev['contradictions_detected']) + ' audit_events=' + str(len(au['events'])))
assert run['final_status'] == 'COMPLETED'
assert ev['contradictions_detected'] == 0
assert run['evidence_coverage'] == 1.0

print()
print('=== SCENARIO 2: False Success ===')
run2 = call('POST', '/workflow/run', {'goal': GOAL, 'max_retries': 3})
inj2 = call('POST', '/workflow/' + run2['workflow_id'] + '/failure-injection',
    {'tool_name':'add_member','mode':'FALSE_SUCCESS','max_fires':1,'max_retries':3,
     'param_filter':{'project_id':'hackathon-alpha'}})
wid2 = inj2['replay_workflow_id']
ev2  = call('GET', '/workflow/' + wid2 + '/evidence')
au2  = call('GET', '/workflow/' + wid2 + '/audit')
print('  status=' + inj2['final_status'] + ' contradictions=' + str(ev2['contradictions_detected']) + ' recovery_events=' + str(au2['summary']['recovery_events']))
assert inj2['final_status'] == 'COMPLETED'
assert ev2['contradictions_detected'] >= 1

print()
print('=== SCENARIO 3: Unrecoverable ===')
run3 = call('POST', '/workflow/run', {'goal': GOAL})
inj3 = call('POST', '/workflow/' + run3['workflow_id'] + '/failure-injection',
    {'tool_name':'create_project','mode':'EXECUTION_FAILURE','max_fires':0,
     'max_retries':2,'max_replan_cycles':0,'param_filter':{'project_id':'hackathon-alpha'}})
wid3 = inj3['replay_workflow_id']
wf3  = call('GET', '/workflow/' + wid3)
print('  status=' + inj3['final_status'] + ' blocked_reason=' + str(wf3['blocked_reason'])[:60])
assert inj3['final_status'] == 'BLOCKED'
assert wf3['blocked_reason']

print()
print('=== SCENARIO 4: Atomic Rollback ===')
run4 = call('POST', '/workflow/run', {'goal': GOAL})
inj4 = call('POST', '/workflow/' + run4['workflow_id'] + '/failure-injection',
    {'tool_name':'set_permission','mode':'EXECUTION_FAILURE','max_fires':0,
     'max_retries':0,'max_replan_cycles':0,'atomic_completion':True,
     'param_filter':{'project_id':'hackathon-alpha'}})
wid4 = inj4['replay_workflow_id']
wf4  = call('GET', '/workflow/' + wid4)
act4 = call('GET', '/workflow/' + wid4 + '/actions')
rb   = sum(1 for a in act4 if a['final_status'] == 'ROLLED_BACK')
print('  status=' + inj4['final_status'] + ' rolled_back=' + str(wf4['rolled_back_actions']))
assert inj4['final_status'] == 'ROLLED_BACK'
assert wf4['rolled_back_actions'] >= 1

print()
print('SUCCESS: All four scenarios verified through the API.')
