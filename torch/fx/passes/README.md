# torch/fx/passes/

Directory layout:

```
.
├── conversion                      # Passes that convert between dialects
├── dialect                         # Passes that are dialect-specific
├── infra                           # Pass Infrastructure
│   ├── pass_base.py                # Base class for creating passes
│   ├── pass_manager.py             # Collection of passes
│   └── pass_pipeline_manager.py    # Collection of pass managers
├── utils                           # Selector and Actor utils
└── README.md
```

Tests can be found in `pytorch/test/fx/`