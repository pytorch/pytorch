import base64
import bz2
import json


def send_to_scribe(logs: str) -> str:
    return _send_to_scribe_via_boto3(logs)


def _send_to_scribe_via_boto3(logs: str) -> str:
    # lazy import so that we don't need to introduce extra dependencies
    import boto3  # type: ignore[import]

    print("Scribe access token not provided, sending report via boto3...")
    event = {"base64_bz2_logs": base64.b64encode(bz2.compress(logs.encode())).decode()}
    client = boto3.client("lambda")
    res = client.invoke(FunctionName='gh-ci-scribe-proxy', Payload=json.dumps(event).encode())
    payload = str(res['Payload'].read().decode())
    if res['FunctionError']:
        raise Exception(payload)
    return payload

if __name__ == '__main__':
    print(send_to_scribe(json.dumps({'invalid': 'logs'})))