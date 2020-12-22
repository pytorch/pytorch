-- All operators with 4d shapes in inputs and outputs
SELECT DISTINCT(t1.operator_name) FROM usage as t1
    JOIN usage as t2 ON t1.operation_id = t2.operation_id AND t1.type = 'args' AND t2.type = 'res'
    WHERE t1.dim = 4 AND t2.dim = 4;

-- Example of join
SELECT t1.*, t2.* FROM usage as t1
    JOIN usage as t2 ON t1.operation_id = t2.operation_id AND t1.type = 'args' AND t2.type = 'res'
    WHERE t1.dim = 4 AND t2.dim = 4 
    LIMIT 1;

-- All operators with channels_last input and channels_last output
SELECT DISTINCT(t1.operator_name) FROM usage as t1
    JOIN usage as t2 ON t1.operation_id = t2.operation_id AND t1.type = 'args' AND t2.type = 'res'
    WHERE t1.dim = 4 AND t1.contiguous = false AND t1.channels_last = true
    AND t2.dim = 4 AND t2.contiguous = false AND t2.channels_last = true;

-- Example of join
SELECT t1.*, t2.* FROM usage as t1
    JOIN usage as t2 ON t1.operation_id = t2.operation_id AND t1.type = 'args' AND t2.type = 'res'
    WHERE t1.dim = 4 AND t2.dim = 4 AND t1.operator_name = 'mean'
    LIMIT 1;

-- By type
SELECT DISTINCT(dtype) FROM usage;

-- All captured operators
SELECT DISTINCT(operator_name) FROM usage;

-- All operators with bfloat16 in inputs and outputs
CREATE TABLE bfloat16 AS
SELECT DISTINCT(t1.operator_name) FROM usage as t1
    JOIN usage as t2 ON t1.operation_id = t2.operation_id AND t1.type = 'args' AND t2.type = 'res'
    WHERE t1.dtype = 'torch.bfloat16' AND t2.dtype = 'torch.bfloat16';