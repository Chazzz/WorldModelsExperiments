for i in `seq 1 16`;
do
  echo worker $i
  # on cloud:
  python extract.py &
  # on macbook for debugging:
  #python extract.py &
  sleep 1.0
done
