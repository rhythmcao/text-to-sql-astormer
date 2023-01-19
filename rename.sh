#!/bin/bash

ori_name=$1
mv data/spider/tables.json data/spider/tables.${ori_name}.json
mv data/spider/train.json data/spider/train.${ori_name}.json
mv data/spider/dev.json data/spider/dev.${ori_name}.json
mv data/sparc/tables.json data/sparc/tables.${ori_name}.json
mv data/sparc/train.json data/sparc/train.${ori_name}.json
mv data/sparc/dev.json data/sparc/dev.${ori_name}.json
mv data/cosql/tables.json data/cosql/tables.${ori_name}.json
mv data/cosql/sql_state_tracking/cosql_train.json data/cosql/sql_state_tracking/cosql_train.${ori_name}.json
mv data/cosql/sql_state_tracking/cosql_dev.json data/cosql/sql_state_tracking/cosql_dev.${ori_name}.json

new_name=$2
#cp data/spider/tables.${new_name}.json data/spider/tables.json
#cp data/spider/train.${new_name}.json data/spider/train.json
#cp data/spider/dev.${new_name}.json data/spider/dev.json
#cp data/sparc/tables.${new_name}.json data/sparc/tables.json
#cp data/sparc/train.${new_name}.json data/sparc/train.json
#cp data/sparc/dev.${new_name}.json data/sparc/dev.json
#cp data/cosql/tables.${new_name}.json data/cosql/tables.json
#cp data/cosql/sql_state_tracking/cosql_train.${new_name}.json data/cosql/sql_state_tracking/cosql_train.json
#cp data/cosql/sql_state_tracking/cosql_dev.${new_name}.json data/cosql/sql_state_tracking/cosql_dev.json
mv data/spider/tables.${new_name}.json data/spider/tables.json
mv data/spider/train.${new_name}.json data/spider/train.json
mv data/spider/dev.${new_name}.json data/spider/dev.json
mv data/sparc/tables.${new_name}.json data/sparc/tables.json
mv data/sparc/train.${new_name}.json data/sparc/train.json
mv data/sparc/dev.${new_name}.json data/sparc/dev.json
mv data/cosql/tables.${new_name}.json data/cosql/tables.json
mv data/cosql/sql_state_tracking/cosql_train.${new_name}.json data/cosql/sql_state_tracking/cosql_train.json
mv data/cosql/sql_state_tracking/cosql_dev.${new_name}.json data/cosql/sql_state_tracking/cosql_dev.json
