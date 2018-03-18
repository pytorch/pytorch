from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import *

Base = declarative_base()


class PerfTestBaseline(Base):
    __tablename__ = 'perf_test_baseline'

    id = Column(String, primary_key=True)
    commit_id = Column(String)
    test_name = Column(String)
    test_type = Column(String)
    mean = Column(Float)
    sigma = Column(Float)


class LatestTestedCommit(Base):
    __tablename__ = 'latest_tested_commit'

    commit_id = Column(String, primary_key=True)
    test_type = Column(String)
