GH_PR_REVIEWS_FRAGMENT = """
fragment PRReviews on PullRequestReviewConnection {
  nodes {
    author {
      login
    }
    state
  }
  pageInfo {
    startCursor
    hasPreviousPage
  }
}
"""

GH_CHECKSUITES_FRAGMENT = """
fragment PRCheckSuites on CheckSuiteConnection {
  edges {
    node {
      app {
        name
        databaseId
      }
      workflowRun {
        workflow {
          name
        }
        url
      }
      checkRuns(first: 50) {
        nodes {
          name
          conclusion
          detailsUrl
        }
        pageInfo {
          endCursor
          hasNextPage
        }
      }
      conclusion
    }
    cursor
  }
  pageInfo {
    hasNextPage
  }
}
"""

GH_COMMIT_AUTHORS_FRAGMENT = """
fragment CommitAuthors on PullRequestCommitConnection {
  nodes {
    commit {
      author {
        user {
          login
        }
        email
        name
      }
      oid
    }
  }
  pageInfo {
    endCursor
    hasNextPage
  }
}
"""

GH_GET_PR_INFO_QUERY = GH_PR_REVIEWS_FRAGMENT + GH_CHECKSUITES_FRAGMENT + GH_COMMIT_AUTHORS_FRAGMENT + """
query ($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      closed
      isCrossRepository
      author {
        login
      }
      title
      body
      headRefName
      headRepository {
        nameWithOwner
      }
      baseRefName
      baseRepository {
        nameWithOwner
        isPrivate
        defaultBranchRef {
          name
        }
      }
      mergeCommit {
        oid
      }
      commits_with_authors: commits(first: 100) {
        ...CommitAuthors
        totalCount
      }
      commits(last: 1) {
        nodes {
          commit {
            checkSuites(first: 10) {
              ...PRCheckSuites
            }
            status {
              contexts {
                context
                state
                targetUrl
              }
            }
            pushedDate
            oid
          }
        }
      }
      changedFiles
      files(first: 100) {
        nodes {
          path
        }
        pageInfo {
          endCursor
          hasNextPage
        }
      }
      reviews(last: 100) {
        ...PRReviews
      }
      comments(last: 5) {
        nodes {
          bodyText
          createdAt
          author {
            login
          }
          authorAssociation
          editor {
            login
          }
          databaseId
        }
        pageInfo {
          startCursor
          hasPreviousPage
        }
      }
      labels(first: 100) {
        edges {
          node {
            name
          }
        }
      }
    }
  }
}
"""

GH_GET_PR_NEXT_FILES_QUERY = """
query ($owner: String!, $name: String!, $number: Int!, $cursor: String!) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      files(first: 100, after: $cursor) {
        nodes {
          path
        }
        pageInfo {
          endCursor
          hasNextPage
        }
      }
    }
  }
}
"""

GH_GET_PR_NEXT_CHECKSUITES = GH_CHECKSUITES_FRAGMENT + """
query ($owner: String!, $name: String!, $number: Int!, $cursor: String!) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      commits(last: 1) {
        nodes {
          commit {
            oid
            checkSuites(first: 10, after: $cursor) {
              ...PRCheckSuites
            }
          }
        }
      }
    }
  }
}
"""

GH_GET_COMMIT_CHECKSUITES = GH_CHECKSUITES_FRAGMENT + """
query ($owner: String!, $name: String!, $commit: String) {
  repository(name: $name, owner: $owner) {
    object(expression: $commit) {
      ... on Commit {
        checkSuites {
          ...PRCheckSuites
        }
      }
    }
  }
}
"""

GH_GET_COMMIT_NEXT_CHECKSUITES = GH_CHECKSUITES_FRAGMENT + """
query ($owner: String!, $name: String!, $commit: String, $cursor: String!) {
  repository(name: $name, owner: $owner) {
    object(expression: $commit) {
      ... on Commit {
        oid
        checkSuites(first: 10, after: $cursor) {
          ...PRCheckSuites
        }
      }
    }
  }
}
"""

GH_GET_COMMIT_NEXT_CHECK_RUNS = """
query ($owner: String!, $name: String!, $cs_cursor: String, $cr_cursor: String!, $commit: String) {
  repository(name: $name, owner: $owner) {
    object(expression: $commit) {
      ... on Commit {
        oid
        checkSuites(first: 1, after: $cs_cursor) {
          nodes {
            checkRuns(first: 100, after: $cr_cursor) {
              nodes {
                name
                conclusion
                detailsUrl
              }
              pageInfo {
                endCursor
                hasNextPage
              }
            }
          }
        }
      }
    }
  }
}
"""

GH_GET_PR_NEXT_CHECK_RUNS = """
query ($owner: String!, $name: String!, $number: Int!, $cs_cursor: String, $cr_cursor: String!) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      commits(last: 1) {
        nodes {
          commit {
            oid
            checkSuites(first: 1, after: $cs_cursor) {
              nodes {
                checkRuns(first: 100, after: $cr_cursor) {
                  nodes {
                    name
                    conclusion
                    detailsUrl
                  }
                  pageInfo {
                    endCursor
                    hasNextPage
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
"""

GH_GET_PR_PREV_COMMENTS = """
query ($owner: String!, $name: String!, $number: Int!, $cursor: String!) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      comments(last: 100, before: $cursor) {
        nodes {
          bodyText
          createdAt
          author {
            login
          }
          authorAssociation
          editor {
            login
          }
          databaseId
        }
        pageInfo {
          startCursor
          hasPreviousPage
        }
      }
    }
  }
}
"""

# This query needs read-org permission
GH_GET_TEAM_MEMBERS_QUERY = """
query($org: String!, $name: String!, $cursor: String) {
  organization(login: $org) {
    team(slug: $name) {
      members(first: 100, after: $cursor) {
        nodes {
          login
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
    }
  }
}
"""

GH_GET_PR_NEXT_AUTHORS_QUERY = GH_COMMIT_AUTHORS_FRAGMENT + """
query ($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      commits_with_authors: commits(first: 100, after: $cursor) {
        ...CommitAuthors
      }
    }
  }
}
"""

GH_GET_PR_PREV_REVIEWS_QUERY = GH_PR_REVIEWS_FRAGMENT + """
query ($owner: String!, $name: String!, $number: Int!, $cursor: String!) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      reviews(last: 100, before: $cursor) {
        ...PRReviews
      }
    }
  }
}
"""
