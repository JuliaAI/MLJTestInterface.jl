classifiers = [
    ConstantClassifier,
    DeterministicConstantClassifier,
]

expected_report1 = (
    name = "ConstantClassifier",
    package_name = "MLJModels",
    model_type = "✓",
    model_instance = "✓",
    fitted_machine = "✓",
    operations = "predict",
)

expected_report2 = (
    name = "DeterministicConstantClassifier",
    package_name = "MLJModels",
    model_type = "✓",
    model_instance = "✓",
    fitted_machine = "✓",
    operations = "predict",
)

@testset "test classifiers on valid data (models are types)" begin

    classifiers = [ConstantClassifier, DeterministicConstantClassifier]
    X, y =  MLJTestInterface.make_binary();

    fails, report  =
        @test_logs MLJTestInterface.test(
            classifiers,
            X,
            y;
            mod=@__MODULE__,
            level=2,
            verbosity=0
        )
    @test isempty(fails)
    @test report[1] == expected_report1
    @test report[2] == expected_report2
end

@testset "test classifiers on invalid data" begin
    X, y = MLJTestInterface.make_regression(); # wrong data for a classifier
    fails, report = @test_logs(
        (:error, r""),
        match_mode=:any,
        MLJTestInterface.test(
            classifiers,
            X,
            y;
            mod=@__MODULE__,
            level=2,
            verbosity=0
        )
    )

    @test length(fails) === 1
    @test fails[1].exception isa ErrorException
    @test merge(fails[1], (; exception="")) == (
        name = "ConstantClassifier",
        package_name = "MLJModels",
        test = "fitted_machine",
        exception = ""
    )

    @test report[1] == (
        name = "ConstantClassifier",
        package_name = "MLJModels",
        model_type = "✓",
        model_instance = "✓",
        fitted_machine = "×",
        operations = "-",
    )

    @test report[2] == (
        name = "DeterministicConstantClassifier",
        package_name = "MLJModels",
        model_type = "✓",
        model_instance = "✓",
        fitted_machine = "✓",
        operations = "predict",
    )

    # throw=true:
    @test_logs(
        (:error, r""), match_mode=:any,
        @test_throws(
            ErrorException,
            MLJTestInterface.test(
                classifiers,
                X,
                y;
                mod=@__MODULE__,
                level=2,
                throw=true,
                verbosity=0
            )
        )
    )
end

classifiers = [ConstantClassifier,]
X, y = MLJTestInterface.make_binary()

@testset "verbose logging" begin
    # progress meter:
    @test_logs MLJTestInterface.test(
        fill(classifiers[1], 500),
        X,
        y;
        mod=@__MODULE__,
        level=1,
        verbosity=1);

    # verbosity high:
    @test_logs(
        (:info, r"Testing ConstantClassifier"),
        (:info, r"model_type"),
        (:info, r"model_instance"),
        (:info, r"fitted_machine"),
        (:info, r"operations"),
        MLJTestInterface.test(
            classifiers,
            X,
            y;
            mod=@__MODULE__,
            level=2,
            verbosity=2)
    )
end

@testset "level" begin
    # level=1:
    fails, report  =
        @test_logs MLJTestInterface.test(
            classifiers,
            X,
            y;
            mod=@__MODULE__,
            level=1,
            verbosity=0)
    @test isempty(fails)
    @test report[1] == (
        name = "ConstantClassifier",
        package_name = "MLJModels",
        model_type = "✓",
        model_instance = "-",
        fitted_machine = "-",
        operations = "-",
    )

    # level=2:
    fails, report  =
        @test_logs MLJTestInterface.test(
            classifiers,
            X,
            y;
            mod=@__MODULE__,
            level=2,
            verbosity=0)
    @test isempty(fails)
    @test report[1] == (
        name = "ConstantClassifier",
        package_name = "MLJModels",
        model_type = "✓",
        model_instance = "✓",
        fitted_machine = "✓",
        operations = "predict",
    )
end


true
