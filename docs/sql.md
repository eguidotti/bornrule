# Born's Classifier SQL

```py
from bornrule.sql import BornClassifierSQL
```

!!! warning 
    
    This SQL implementation is compatible with `PostgreSQL 12+`, `SQLite v3.24.0+`, and `MySQL 9.4.0+`. 
    Earlier versions of `PostgreSQL` and `MySQL` may also work, but they have not been tested. Versions of `SQLite` earlier than `3.24.0` are not supported.

::: bornrule.sql.BornClassifierSQL
    options:
      members:
        - get_params
        - set_params
        - fit
        - partial_fit
        - predict
        - predict_proba
        - explain
        - deploy
        - undeploy
        - is_fitted
        - is_deployed
