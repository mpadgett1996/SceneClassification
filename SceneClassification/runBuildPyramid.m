% function runBuildPyramid(imageBaseDir)

    % if vlfeat package is not installed, vlroc() will not function
    % run /Volumes/Lexar/CS385_OR/vlfeat/toolbox/vl_setup

    % Building the pyramids will generate lots of warnings!
    % To disable them in later runs, use: warning('off', 'last');


    %% % CONSTANTS - DO NOT CHANGE %%%
    % train/test category-index constants
    CAT_TRAIN = 1;
    CAT_TEST  = 2;
    % positive/negative class labels
    LBL_POS     =  1;
    LBL_NEG     = -1;
    LBL_NEG_ROC =  0;

    %%%  VARIABLES - these you may change  %%%
    % fraction of images to use as training
    train_test_ratio = 1/2;

    kernel_type = 'rbf';

    pyramid_params.maxImageSize = 1000;
    pyramid_params.gridSpacing = 8;
    pyramid_params.patchSize = 16;
    pyramid_params.dictionarySize = 200;
    pyramid_params.numTextonImages = 150; %50 was default
    pyramid_params.pyramidLevels = 1; %3 was default
    pyramid_params.oldSift = false;
    

    imageBaseDir = '/Users/student/Documents/MATLAB/Spatial_Pyramid/images/';
    subdir_list = {
        'airport', ...
        'auditorium', ...
        'bamboo_forest', ...
        'campus', ...
        'desert', ...
        'football_field', ...
        'kitchen', ...
        'sky'};
    
    % seed random number generator (any number will do)
%     rng(47);
    
%% DIRECTORY MAPPING

    filenames_train = {};
    filenames_test  = {};
    filenum = [0, 0];
    % number of train|test images in each category//subdirectory
    category_num = zeros(2, length(subdir_list));

    for s = 1:length(subdir_list)
        imagestruct = dir(strcat(imageBaseDir,subdir_list{s}));
        imagestruct([imagestruct.isdir]) = []; % remove any directories

        num_images = length(imagestruct);
        division_point = floor(train_test_ratio * num_images);
        perm = uint32(randperm(num_images));

        category_num(CAT_TRAIN, s) = division_point;
        category_num(CAT_TEST,  s) = num_images-division_point;

        % first part for training
        for i = 1:division_point
            filenum(CAT_TRAIN) = filenum(CAT_TRAIN) + 1;
            filenames_train{filenum(CAT_TRAIN)} = [subdir_list{s} '/' imagestruct(perm(i)).name];
        end
        % second part for testing
        for i = division_point+1:num_images
            filenum(CAT_TEST)  = filenum(CAT_TEST)  + 1;
            filenames_test{filenum(CAT_TEST)}  = [subdir_list{s} '/' imagestruct(perm(i)).name];
        end
    end

    %% PYRAMID FORMATION
    pyramid_train = BuildPyramid(filenames_train, imageBaseDir, strcat(imageBaseDir, 'descriptors'), pyramid_params);
    pyramid_test  = BuildPyramid(filenames_test,  imageBaseDir, strcat(imageBaseDir, 'descriptors'), pyramid_params);


    %% SVM FORMATION
    % svm needs: interest point files, labels (pos/neg)
    % need to associate pos/neg class labels for each row of pyramid_all for training

    train_labels = LBL_NEG * ones(length(subdir_list), sum(category_num(CAT_TRAIN,:)));
    test_labels  = LBL_NEG * ones(length(subdir_list), sum(category_num(CAT_TEST, :)));
    svmModels = cell(length(subdir_list), 1);

    for subdir = 1:length(subdir_list)
        prior_images = [0, 0];
        if subdir > 1
            prior_images(CAT_TRAIN) = sum(category_num(CAT_TRAIN, 1:subdir-1));
            prior_images(CAT_TEST)  = sum(category_num(CAT_TEST,  1:subdir-1));
        end

        % set up truth labels
        img_num_train = category_num(CAT_TRAIN, subdir);
        img_num_test  = category_num(CAT_TEST,  subdir);
        train_labels(subdir, prior_images(CAT_TRAIN)+1: prior_images(CAT_TRAIN)+img_num_train) = LBL_POS;
        test_labels (subdir, prior_images(CAT_TEST) +1: prior_images(CAT_TEST) +img_num_test)  = LBL_POS;

        % SVM CREATION
        svmModels{subdir} = fitcsvm(pyramid_train, train_labels(subdir,:)', 'KernelFunction', kernel_type, ...
            'Standardize', true, 'KernelScale', 'auto', 'Classnames', [1, -1]);
    end

    % roc curves as used in bag-of-words assignment needs a different value for negative labels
    train_labels_roc = train_labels;
    train_labels_roc(train_labels == LBL_NEG) = LBL_NEG_ROC;
    test_labels_roc  = test_labels;
    test_labels_roc (test_labels  == LBL_NEG) = LBL_NEG_ROC;


    %% TRAINING ACCURACY
    % hold score of predictions on each image for all models
    all_scores_training = zeros(filenum(CAT_TRAIN), length(svmModels));

    all_roc_curves_train = zeros(length(subdir_list), length(filenames_train), 2);
    all_rpc_curves_train = zeros(length(subdir_list), length(filenames_train), 2);
    
    % for each subdir, run training data through the model trained to separate it from the rest
    for subdir = 1:length(subdir_list)
        [predict_train,score_training] = predict(svmModels{subdir}, pyramid_train);
        positive_rate = score_training(:,1);
        all_scores_training(:,subdir) = positive_rate;

        correct_train = sum(predict_train' == train_labels(subdir,:));
        fprintf('Training SVM %d: correct predictions: %d / %d\n', subdir, correct_train, length(train_labels(subdir,:)));

        %%% compute vl-roc, roc
        [~, ~, info] = vl_roc(train_labels(subdir,:), positive_rate');
        [roc_curve_train,roc_op_train,roc_area_train,roc_threshold_train] = roc([positive_rate, train_labels_roc(subdir,:)']);
        all_roc_curves_train(subdir,:,:) = roc_curve_train;
        fprintf('Training SVM %d: Area under ROC curve = %f; Optimal threshold = %f\n', subdir, roc_area_train, roc_threshold_train);
        fprintf('Training SVM %d: AreaUnderCurve VL_ROC: %f\n', subdir, info.auc);
        %%% compute rpc
        [rpc_curve_train,rpc_ap_train,rpc_area_train,rpc_threshold_train] = ...
            recall_precision_curve([positive_rate, train_labels_roc(subdir,:)'], category_num(CAT_TRAIN, subdir));
        all_rpc_curves_train(subdir,:,:) = rpc_curve_train;
        fprintf('Training SVM %d: Area under RPC curve = %f\n', subdir, rpc_area_train);

    end

    %% TESTING ACCURACY
    % hold score of predictions on each testing image for all models
    all_scores_testing = zeros(filenum(CAT_TEST), length(svmModels));
    all_roc_curves_test  = zeros(length(subdir_list), length(filenames_test ), 2);
    all_rpc_curves_test  = zeros(length(subdir_list), length(filenames_test ), 2);

    % for each subdir, run testing data through the model trained to separate it from the rest
    for subdir = 1:length(subdir_list)
        [predict_test, score_testing] = predict(svmModels{subdir}, pyramid_test);
        positive_rate = score_testing(:, 1);
        all_scores_testing(:,subdir) = positive_rate;

        correct_test = sum(predict_test' == test_labels(subdir,:));
        fprintf('Testing SVM %d:  correct predictions: %d / %d\n', subdir, correct_test, length(test_labels(subdir,:)));

        %%% compute vl-roc, roc
        [tpr, tnr, info] = vl_roc(test_labels(subdir,:), positive_rate');
        [roc_curve_test,roc_op_test,roc_area_test,roc_threshold_test] = roc([positive_rate, test_labels_roc(subdir,:)']);
        all_roc_curves_test(subdir,:,:) = roc_curve_test;
        fprintf('Testing SVM %d:  Area under ROC curve = %f\n', subdir, roc_area_test);
        fprintf('Testing SVM %d:  AreaUnderCurve VL_ROC: %f\n', subdir, info.auc);
        %%% compute rpc
        [rpc_curve_test,rpc_ap_test,rpc_area_test,rpc_threshold_test] = ...
            recall_precision_curve([positive_rate, test_labels_roc(subdir,:)'], category_num(CAT_TEST,subdir));
        all_rpc_curves_test(subdir,:,:) = rpc_curve_test;
        fprintf('Testing SVM %d:  Area under RPC curve = %f\n', subdir, rpc_area_test);

    end

    %% MODEL TRAIN EVALUATION
    % scores:
    %	    svmA  svmB  svmC  ...
    % img1	40%   32%   61%   ...
    % img2	...

    [~, choice_train] = max(all_scores_training, [], 2);
    [~, choice_test]  = max(all_scores_testing,  [], 2);

    truths_train = 1:length(subdir_list);
    truths_train = repelem(truths_train, category_num(CAT_TRAIN,:))';
    truths_test  = 1:length(subdir_list);
    truths_test  = repelem(truths_test,  category_num(CAT_TEST, :))';

    confusion_train = confusionmat(truths_train, choice_train);
    confusion_test  = confusionmat(truths_test,  choice_test );
    % normalize confusion matrix
    for r = 1:length(confusion_train)
        confusion_train(r, :) = (confusion_train(r,:)/category_num(CAT_TRAIN, r));
        confusion_test (r, :) = (confusion_test (r,:)/category_num(CAT_TEST,  r));
    end

    fprintf('Accuracy of training predictions for each class: [%s ]\n', sprintf('% 4.2f%% ',100*diag(confusion_train)'));
    fprintf('Accuracy of testing  predictions for each class: [%s ]\n', sprintf('% 4.2f%% ',100*diag(confusion_test )'));

    %% PLOTTING section - plot some figures to see what is going on...
    
    % Plot Training Confusion Matrix with class names
    confMatTrain = confMatGet(truths_train, choice_train);
    opt=confMatPlot('defaultOpt');
    opt.className={'airport', 'auditorium', 'bamboo forest', 'campus', 'desert', 'football field', 'kitchen', 'sky'};
    
    % Percentage plot
    opt.mode='percentage';  % or 'dataCount' or 'both'
    opt.format='8.2f';
    figure; confMatPlot(confMatTrain, opt);
    
 
    % Plot Testing Confusion Matrix with class names
    confMatTest = confMatGet(truths_test, choice_test);
    opt=confMatPlot('defaultOpt');
    opt.className={'airport', 'auditorium', 'bamboo forest', 'campus', 'desert', 'football field', 'kitchen', 'sky'};

    % Percentage plot
    opt.mode='percentage';
    opt.format='8.2f';
    figure; confMatPlot(confMatTest, opt);
    

    % PLOT BOTH TRAINING AND TESTING TOGETHER

    figure; 
    subplot(1, 2, 1); hold on;
    title('Confusion Matrix - Training');
    confMatPlot(confMatTrain, opt);

    subplot(1, 2, 2); hold on;
    title('Confusion Matrix - Testing');
    confMatPlot(confMatTest, opt);
    
    
    %%
    % write confusion matrices out for use in python heatmap generator
    %  (python2.7, requires numpy module)
%     csvwrite(strcat(imageBaseDir, 'confusion_train.csv'), confusion_train);
%     csvwrite(strcat(imageBaseDir, 'confusion_test.csv' ), confusion_test );

    % ROC plotting
%     Look at the classification performance
    

    for subdir = 1:length(subdir_list)
        figure;hold on;
%         subplot(1, length(subdir_list), subdir);
        plot(all_roc_curves_train(subdir,:,1),all_roc_curves_train(subdir,:,2),'r');
        plot(all_roc_curves_test(subdir,:,1),all_roc_curves_test(subdir,:,2),'g');
        axis([0 1 0 1]); axis square; grid on;
        xlabel('P_{fa}'); ylabel('P_d');
        title(sprintf('ROC Curve, SVM %d -  %s', subdir, kernel_type));
        legend('Train','Test');
        hold off;
    end

    % Look at the retrieval performance
    for subdir = 1:length(subdir_list)
        figure;hold on;
%         subplot(1, length(subdir_list), subdir);
        plot(rpc_curve_train(:,1),rpc_curve_train(:,2),'r');
        plot(rpc_curve_test(:,1),rpc_curve_test(:,2),'g');
        axis([0 1 0 1]); axis square; grid on;
        xlabel('Recall'); ylabel('Precision');
        title(sprintf('RPC Curve, SVM %d -  %s', subdir, kernel_type));
        legend('Train','Test');
    end


    %% STATS (image train|test and category division, etc.)
    fprintf('Source file breakdown:\n');
    for s = 1:length(category_num)
        fprintf('\t%4d training, %4d testing from %s\n', category_num(CAT_TRAIN,s), category_num(CAT_TEST,s), subdir_list{s});
    end
    fprintf('Totals:\t%4d training, %4d testing, %d overall\n', ...
        sum(category_num(CAT_TRAIN,:)), sum(category_num(CAT_TEST,:)), sum(sum(category_num)));


