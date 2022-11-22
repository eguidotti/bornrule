# Born's Classifier SQL

!!! warning 
    
    This SQL implementation is in beta release. 
    It is compatible with `SQLite v3.24.0+` and `PostgreSQL 14`. 
    Previous versions of `PostgreSQL` may also work, but they have not been tested.

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
