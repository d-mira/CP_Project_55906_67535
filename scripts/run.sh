#!/bin/bash

#Tests log file
LOG_FILE="output.log"

compile(){
  echo "Compiling application... 📦️"
  echo "Compiling sequential version..."
  mkdir -p ../build-seq
  cd ../build-seq
  cmake -DRUN_SEQUENTIAL=ON ..
  make
  cd ../scripts
  echo "Sequential version done. ✅"
  echo "Compiling parallel version..."
  mkdir -p ../build-par
  cd ../build-par
  cmake -DRUN_PARALLEL=ON ..
  make
  cd ../scripts
  echo "Parallel version done. ✅"
  echo "Compiling CUDA version..."
  mkdir -p ../build-cub
  cd ../build-cub
  cmake -DRUN_CUDA=ON ..
  make
  cd ../scripts
  echo "CUDA version done. ✅"

  echo "Compilation completed 👍️"
  sleep 3
  #clear
}

check_compiled() {
    #if [[ ! -f "../build-seq/project" || ! -f "../build-par/project_par" ]]; then
    if [[ ! -f "build-seq/project" || ! -f "build-par/project_par" || ! -f "build-cub/project_cub" ]]; then
        echo "One or more applications are not compiled."
        read -p "Do you want to compile them now? (y/n): " choice
        if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
            compile
        else
            echo "Cannot proceed without compilation."
            echo "Exiting."
            sleep 0.5
            echo "Exiting.."
            sleep 0.5
            echo "Exiting..."
            sleep 0.5
            echo "bye 👋"
            exit 1
        fi
    fi
}

run_tests() {
    clear
    echo "Running tests..." | tee -a $LOG_FILE
    sleep 1

    # Test sequential version
    ./../build-seq/project ../dataset/sample_5184×3456.ppm 1 ../outputDataset/sample_5184×3456_1.ppm | tee -a $LOG_FILE
    sleep 1
    #./../build-seq/project ../dataset/sample_5184×3456.ppm 100 ../outputDataset/sample_5184×3456_100.ppm | tee -a $LOG_FILE
    sleep 1
    ./../build-seq/project ../dataset/sample_5184×3456.ppm 500 ../outputDataset/sample_5184×3456_500.ppm | tee -a $LOG_FILE
    sleep 1
    ./../build-seq/project ../dataset/sample_5184×3456.ppm 1000 ../outputDataset/sample_5184×3456_1000.ppm | tee -a $LOG_FILE
    sleep 1

    # Test parallel version
    ./../build-par/project_par ../dataset/sample_5184×3456.ppm 1 ../outputDataset/sample_5184×3456_1.ppm | tee -a $LOG_FILE
    sleep 1
    ./../build-par/project_par ../dataset/sample_5184×3456.ppm 100 ../outputDataset/sample_5184×3456_100.ppm | tee -a $LOG_FILE
    sleep 1
    ./../build-par/project_par ../dataset/sample_5184×3456.ppm 500 ../outputDataset/sample_5184×3456_500.ppm | tee -a $LOG_FILE
    sleep 1
    ./../build-par/project_par ../dataset/sample_5184×3456.ppm 1000 ../outputDataset/sample_5184×3456_1000.ppm | tee -a $LOG_FILE
    sleep 1

     Test CUDA version
    ./../build-cub/project_cub ../dataset/sample_5184×3456.ppm 1 sample_5184×3456_1.ppm | tee -a $LOG_FILE
    sleep 1
    ./../build-cub/project_cub ../dataset/sample_5184×3456.ppm 100 sample_5184×3456_100.ppm | tee -a $LOG_FILE
    sleep 1
    ./../build-cub/project_cub ../dataset/sample_5184×3456.ppm 500 sample_5184×3456_500.ppm | tee -a $LOG_FILE
    sleep 1
    ./../build-cub/project_cub ../dataset/sample_5184×3456.ppm 1000 sample_5184×3456_1000.ppm | tee -a $LOG_FILE

    echo "Tests completed. All tests logged in output.log. 👍️" | tee -a $LOG_FILE
}

#MENU
check_compiled
echo "Iterative Histogram Equalization"
echo ""
echo "Select an option:"
echo "1. Run tests"
echo "2. Run the application normally"
echo "3. Exit"

read -p "Enter your choice: " choice


case $choice in

    1)
        check_compiled
        run_tests
        ;;
    2)
        check_compiled
        clear
        echo "Running the application normally..."
        read -p "Enter the input image path: " input_image
        read -p "Enter the number of iterations: " iterations
        read -p "Enter the output image (or enter the path to one of the images inside outpuDataset to compare): " output_image
        echo "Select the version to run:"
        echo "1. Sequential"
        echo "2. Parallel"
        echo "3. CUDA"
        read -p "Enter your choice: " version_choice

        case $version_choice in
            1)
                ./../build-seq/project "$input_image" "$iterations" "$output_image"
                ;;
            2)
                ./../build-par/project_par "$input_image" "$iterations" "$output_image"
                ;;
            3)
                ./../build-cub/project_cub "$input_image" "$iterations" "$output_image"
                ;;
            *)
                echo "Invalid choice. Please enter a valid option."
                ;;
        esac
        ;;
    3)
        echo "Exiting the script."
        exit
        ;;
    *)
        echo "Invalid choice. Please enter a valid option."
        ;;
esac
