import argparse
import git
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db_schema import Base, PerfTestBaseline, LatestTestedCommit
import json

parser = argparse.ArgumentParser()
parser.add_argument('--username', dest='username', action='store',
                    required=False, help='username for database')
parser.add_argument('--password', dest='password', action='store',
                    required=False, help='password for database')
parser.add_argument('--hostname', dest='hostname', action='store',
                    required=False, help='hostname for database')
parser.add_argument('--dbname', dest='dbname', action='store',
                    required=False, help='dbname for database')
parser.add_argument('--testtype', dest='testtype', action='store',
                    required=True, help='type of perf test')
parser.add_argument('--datafile', dest='datafile', action='store',
                    required=True, help='file to store the baseline numbers in')
parser.add_argument('--local', dest='local', action='store_true',
                    required=False, help='local run')
args = parser.parse_args()

if args.local:
    engine = create_engine('sqlite:///test.db')
else:
    engine = create_engine('postgresql://{}:{}@{}/{}'.format(args.username, args.password, args.hostname, args.dbname))

Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

data = {}

latest_tested_commit = session.query(LatestTestedCommit).filter_by(test_type=args.testtype).first()
if latest_tested_commit:
    latest_tested_commit_id = latest_tested_commit.commit_id
    data['commit'] = latest_tested_commit_id
    baseline_list = session.query(PerfTestBaseline) \
                           .filter_by(commit_id=latest_tested_commit_id, test_type=args.testtype) \
                           .all()
    for baseline in baseline_list:
        data[baseline.test_name] = {}
        data[baseline.test_name]['mean'] = baseline.mean
        data[baseline.test_name]['sigma'] = baseline.sigma

with open(args.datafile, 'w') as new_data_file:
    json.dump(data, new_data_file, indent=4)
