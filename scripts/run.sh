#!/bin/bash

compile(){
  echo "Compiling application..."
  echo "Compiling sequential version..."
  mkdir -p build-seq
  cd build-seq
  cmake -DRUN_SEQUENTIAL=ON ..
  make
  cd ..
  echo "Compiling parallel version..."
  mkdir -p build-par
  cd build-par
  cmake -DRUN_PARALLEL=ON ..
  make
  cd ..
  echo "Compiling CUDA version..."
  mkdir -p build-cub
  cd build-cub
  cmake -DRUN_CUDA=ON ..
  make
  cd ..

  echo "Compilation completed"
}

check_compiled(){
  if[[ ! -f "build-seq/project" || ! -f "build-par/project_par" || ! -f "build-cub/project_cub"]]; then
    echo "One or more version of the application are not compiled."
    read -p "Do you want to compile now? (y/n)" choice

    if[["$choice" == "y" || "$choice" == "Y"]]; then
      compile
      else
        echo "Cannot proceed without compiling."
        echo "Exiting."
        sleep 1
        echo "Exiting.."
        sleep 1
        echo "Exiting..."
        sleep 1
        echo "bye 👋"
        sleep 1
        exit1
    fi
  fi
}