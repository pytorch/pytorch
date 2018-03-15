import argparse
import git
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db_schema import Base, PerfTestBaseline, LatestTestedCommit
from sqlalchemy.dialects.postgresql import insert
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
                    required=True, help='file that contains the new baseline numbers')
parser.add_argument('--local', dest='local', action='store_true',
                    required=False, help='local run')
args = parser.parse_args()

if args.local:
    engine = create_engine('sqlite:///test.db')
else:
    engine = create_engine('postgresql://{}:{}@{}/{}'.format(args.username, args.password, args.hostname, args.dbname))

Base.metadata.create_all(engine)

if args.datafile:
    Session = sessionmaker(bind=engine)
    session = Session()

    new_baseline_list = []
    old_baseline_list = []
    current_commit_id = None
    with open(args.datafile) as data_file:
        data = json.load(data_file)
        current_commit_id = data['commit']

        # Delete all existing baseline numbers from this commit
        for baseline in session.query(PerfTestBaseline).filter_by(commit_id=current_commit_id).all():
            session.delete(baseline)

        for key in data:
            if 'mean' in data[key] and 'sigma' in data[key]:
                test_name = key
                mean = float(data[key]['mean'])
                sigma = float(data[key]['sigma'])

                new_baseline_list.append(
                    PerfTestBaseline(
                        id=current_commit_id + '_' + args.testtype + '_' + test_name,
                        commit_id=current_commit_id,
                        test_name=test_name,
                        test_type=args.testtype,
                        mean=float(mean),
                        sigma=float(sigma)
                    )
                )

        session.add_all(new_baseline_list)

    latest_tested_commit = session.query(LatestTestedCommit).filter_by(test_type=args.testtype).first()
    if not latest_tested_commit:
        latest_tested_commit = LatestTestedCommit(commit_id=current_commit_id, test_type=args.testtype)
        session.add(latest_tested_commit)
    else:
        latest_tested_commit_id = latest_tested_commit.commit_id

        repo = git.Repo('../../')
        commit_ids = repo.git.rev_list('HEAD').splitlines()

        # if latest_tested_commit_id is in commit history, that means the current commit is newer
        if latest_tested_commit_id in commit_ids:
            latest_tested_commit.commit_id = current_commit_id
            session.add(latest_tested_commit)

    session.commit()
