Levantar imagenes: docker compose -f docker-compose.platform.yml build
Leventar entorno completo: docker compose -f docker-compose.platform.yml up -d
Vericación de servicios: docker ps
Logs por servicio: docker logs mlflow --tail 50 

Borrar y levantar TODO desde cero: 
docker compose -f docker-compose.platform.yml down -v --remove-orphans
docker system prune -a --volumes -f

Verificar que todo se borra:
docker ps -a      # no debería listar nada
docker images     # debería estar vacío
docker volume ls  # debería estar vacío

Subir imagenes: docker compose -f docker-compose.platform.yml build --no-cache
Subir todo nuevamente: docker compose -f docker-compose.platform.yml up -d


Borrar ejecuciones del DAG:
docker exec -it airflow-webserver bash
airflow dags list-runs -d ml_pipeline_dag
airflow dags delete -y ml_pipeline_dag

borrar logs:
rm -rf /opt/airflow/logs/ml_pipeline_dag
airflow dags delete -y ml_pipeline_dag

Borrar datos preprocesados:
docker exec -it airflow-webserver bash

rm -rf /opt/airflow/data/processed/*
rm -rf /opt/airflow/data/interim/*

echo '{"last_chunk": 0}' > /opt/airflow/data/state.json
exit
docker restart airflow-webserver airflow-scheduler

reiniciar de otra forma: docker compose -f docker-compose.platform.yml restart airflow-webserver airflow-scheduler

docker compose -f docker-compose.platform.yml restart diabetes-api

- docker compose -f docker-compose.platform.yml build diabetes-api
