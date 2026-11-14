# ramjet

## Best Practices

- Never use `print` unless outputting to stdout. Always use `log` from `logger.py`.
- Never raise an error. Always use `log.error`, then `sys.exit`, instead.
