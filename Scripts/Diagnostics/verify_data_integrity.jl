using DataFrames, CSV, Printf, Dates

println("="^80)
println("🔍 HACKATHON DATA DETECTIVE - Folder: $(pwd())")
println("="^80)
println("Analyzing ALL data files to understand your setup...")
println()

# ============================================================================
# 1. SCAN ALL FILES IN CURRENT FOLDER
# ============================================================================
println("📁 STEP 1: SCANNING ALL FILES")
println("-"^50)

# Get all files
files = readdir(".")
println("Total files: $(length(files))")

# Categorize files
categories = Dict(
    "CSV Files" => [],
    "Python Scripts" => [],
    "Julia Scripts" => [],
    "Model Files" => [],
    "Excel Files" => [],
    "Text Files" => [],
    "Other" => []
)

for file in files
    if endswith(file, ".csv")
        push!(categories["CSV Files"], file)
    elseif endswith(file, ".py")
        push!(categories["Python Scripts"], file)
    elseif endswith(file, ".jl")
        push!(categories["Julia Scripts"], file)
    elseif endswith(file, ".model") || endswith(file, ".onnx")
        push!(categories["Model Files"], file)
    elseif endswith(file, ".xlsx") || endswith(file, ".xls")
        push!(categories["Excel Files"], file)
    elseif endswith(file, ".txt") || endswith(file, ".json")
        push!(categories["Text Files"], file)
    else
        push!(categories["Other"], file)
    end
end

# Print categories
for (category, file_list) in categories
    if !isempty(file_list)
        println("📂 $category ($(length(file_list))):")
        for file in sort(file_list)[1:min(5, length(file_list))]
            println("   • $file")
        end
        if length(file_list) > 5
            println("   • ... and $(length(file_list)-5) more")
        end
    end
end

println()

# ============================================================================
# 2. ANALYZE CSV FILES
# ============================================================================
println("📊 STEP 2: ANALYZING CSV FILES")
println("-"^50)

if !isempty(categories["CSV Files"])
    csv_analysis = []
    
    for csv_file in sort(categories["CSV Files"])
        try
            df = CSV.read(csv_file, DataFrame; limit=3)  # Read just first few rows
            n_rows = try
                CSV.read(csv_file, DataFrame) |> nrow
            catch
                "Unknown"
            end
            
            n_cols = size(df, 2)
            col_names = names(df)
            
            # Detect file type by name and content
            file_type = "Unknown"
            if occursin("feature", lowercase(csv_file))
                file_type = "Feature File"
            elseif occursin("train", lowercase(csv_file))
                file_type = "Training Data"
            elseif occursin("test", lowercase(csv_file))
                file_type = "Test Data"
            elseif occursin("submission", lowercase(csv_file))
                file_type = "Submission File"
            end
            
            push!(csv_analysis, (csv_file, file_type, n_rows, n_cols, col_names[1:min(3, length(col_names))]))
            
        catch e
            push!(csv_analysis, (csv_file, "Unreadable", "Error", "Error", []))
        end
    end
    
    # Print CSV analysis
    println("Found $(length(csv_analysis)) CSV files:")
    println()
    println("File Name                      | Type            | Rows    | Cols | Sample Columns")
    println("-"^80)
    
    for (file, ftype, rows, cols, sample_cols) in csv_analysis
        file_display = length(file) > 30 ? file[1:27] * "..." : file
        cols_display = cols isa Int ? cols : "-"
        rows_display = rows isa Int ? rows : "-"
        
        sample_str = join(sample_cols[1:min(3, length(sample_cols))], ", ")
        if length(sample_cols) > 3
            sample_str *= "..."
        end
        
        println(@sprintf("%-30s | %-15s | %-7s | %-4s | %s", 
            file_display, ftype, rows_display, cols_display, sample_str))
    end
else
    println("No CSV files found!")
end

println()

# ============================================================================
# 3. CHECK FOR COMMON HACKATHON PATTERNS
# ============================================================================
println("🎯 STEP 3: DETECTING COMMON HACKATHON PATTERNS")
println("-"^50)

patterns_found = []

# Pattern 1: Multiple feature file versions
feature_files = filter(x -> occursin("feature", lowercase(x)), categories["CSV Files"])
if length(feature_files) > 1
    push!(patterns_found, "⚠️  MULTIPLE FEATURE FILES: $(join(feature_files, ", "))")
    
    # Check if they have same row counts
    row_counts = Dict{String, Any}()
    for file in feature_files
        try
            df = CSV.read(file, DataFrame)
            row_counts[file] = nrow(df)
        catch
            row_counts[file] = "Error"
        end
    end
    
    # Find duplicates
    for (file1, count1) in row_counts
        for (file2, count2) in row_counts
            if file1 != file2 && count1 == count2 && count1 != "Error"
                push!(patterns_found, "   • $file1 and $file2 both have $count1 rows (possible duplicates)")
            end
        end
    end
end

# Pattern 2: Train/test CSV presence
train_csvs = filter(x -> occursin("train", lowercase(x)) && endswith(x, ".csv"), files)
test_csvs = filter(x -> occursin("test", lowercase(x)) && endswith(x, ".csv"), files)

if !isempty(train_csvs)
    push!(patterns_found, "✅ TRAINING DATA: $(join(train_csvs, ", "))")
else
    push!(patterns_found, "❌ NO TRAINING CSV FOUND")
end

if !isempty(test_csvs)
    push!(patterns_found, "✅ TEST DATA: $(join(test_csvs, ", "))")
else
    push!(patterns_found, "❌ NO TEST CSV FOUND")
end

# Pattern 3: Model files
if !isempty(categories["Model Files"])
    push!(patterns_found, "🤖 MODEL FILES: $(join(categories["Model Files"], ", "))")
end

# Pattern 4: Submission files
submission_files = filter(x -> occursin("submission", lowercase(x)), categories["CSV Files"])
if !isempty(submission_files)
    push!(patterns_found, "📤 SUBMISSION FILES: $(join(submission_files, ", "))")
end

# Print patterns
for pattern in patterns_found
    println(pattern)
end

println()

# ============================================================================
# 4. CHECK DATA CONSISTENCY
# ============================================================================
println("🔗 STEP 4: CHECKING DATA CONSISTENCY")
println("-"^50)

# Look for train.csv and test.csv
train_csv = isempty(train_csvs) ? nothing : train_csvs[1]
test_csv = isempty(test_csvs) ? nothing : test_csvs[1]

if train_csv !== nothing
    try
        train_df = CSV.read(train_csv, DataFrame)
        println("📘 $train_csv: $(nrow(train_df)) rows")
        
        # Check for ID column
        id_cols = filter(x -> occursin("id", lowercase(string(x))), names(train_df))
        if !isempty(id_cols)
            println("   ID column: $(id_cols[1])")
            
            # Sample IDs
            sample_ids = train_df[1:min(3, nrow(train_df)), Symbol(id_cols[1])]
            println("   Sample IDs: $sample_ids")
            
            # Check for .png extension
            first_id = string(sample_ids[1])
            if endswith(first_id, ".png")
                println("   📸 IDs have .png extension")
            end
        end
        
        # Check for label column
        label_cols = filter(x -> occursin("label", lowercase(string(x))) || 
                                 occursin("ground", lowercase(string(x))) ||
                                 occursin("target", lowercase(string(x))), names(train_df))
        if !isempty(label_cols)
            println("   Label column: $(label_cols[1])")
            # Check class balance
            if nrow(train_df) > 0
                labels = train_df[!, Symbol(label_cols[1])]
                unique_labels = unique(labels)
                println("   Unique labels: $unique_labels")
                if length(unique_labels) == 2
                    count_0 = sum(labels .== unique_labels[1])
                    count_1 = sum(labels .== unique_labels[2])
                    println("   Class distribution: $count_0 vs $count_1")
                end
            end
        end
    catch e
        println("❌ Could not read $train_csv: $e")
    end
end

if test_csv !== nothing
    try
        test_df = CSV.read(test_csv, DataFrame)
        println("\n📗 $test_csv: $(nrow(test_df)) rows")
        
        # Check for ID column
        id_cols = filter(x -> occursin("id", lowercase(string(x))), names(test_df))
        if !isempty(id_cols)
            println("   ID column: $(id_cols[1])")
            
            # Sample IDs
            sample_ids = test_df[1:min(3, nrow(test_df)), Symbol(id_cols[1])]
            println("   Sample IDs: $sample_ids")
        end
    catch e
        println("❌ Could not read $test_csv: $e")
    end
end

# Compare train and test sizes
if train_csv !== nothing && test_csv !== nothing
    try
        train_rows = nrow(CSV.read(train_csv, DataFrame))
        test_rows = nrow(CSV.read(test_csv, DataFrame))
        total = train_rows + test_rows
        train_pct = round(100 * train_rows / total, digits=1)
        test_pct = round(100 * test_rows / total, digits=1)
        
        println("\n📈 TRAIN/TEST SPLIT:")
        println("   Training: $train_rows rows ($train_pct%)")
        println("   Testing:  $test_rows rows ($test_pct%)")
        println("   Total:    $total rows")
    catch
        println("\n⚠️  Could not compare train/test sizes")
    end
end

println()

# ============================================================================
# 5. IDENTIFY POTENTIAL ISSUES
# ============================================================================
println("🚨 STEP 5: IDENTIFYING POTENTIAL ISSUES")
println("-"^50)

issues = []

# Issue 1: Too many feature files
if length(feature_files) > 3
    push!(issues, "Too many feature files ($(length(feature_files))) - which one to use?")
end

# Issue 2: Missing key files
key_files = ["train.csv", "test.csv", "transformer_model.onnx"]
for file in key_files
    if !(file in files)
        push!(issues, "Missing key file: $file")
    end
end

# Issue 3: Check feature file consistency
if !isempty(feature_files)
    # Try to find a feature file with "train" in name
    train_feature_files = filter(x -> occursin("train", lowercase(x)), feature_files)
    if !isempty(train_feature_files)
        try
            feat_file = train_feature_files[1]
            df = CSV.read(feat_file, DataFrame)
            
            # Check column structure
            if "image_id" in names(df)
                println("✅ Feature file '$feat_file' has image_id column")
                println("   Rows: $(nrow(df)), Columns: $(ncol(df))")
                
                # Check if it has feature columns
                feature_cols = filter(x -> startswith(string(x), "feature_"), names(df))
                if !isempty(feature_cols)
                    println("   Found $(length(feature_cols)) feature columns")
                else
                    push!(issues, "Feature file '$feat_file' has no feature_* columns")
                end
            else
                push!(issues, "Feature file '$feat_file' missing image_id column")
            end
        catch e
            push!(issues, "Cannot read feature file: $e")
        end
    end
end

# Issue 4: Check submission files
if !isempty(submission_files)
    try
        sub_file = submission_files[1]
        df = CSV.read(sub_file, DataFrame)
        println("📤 Submission file '$sub_file': $(nrow(df)) rows")
        
        # Check columns
        if "id" in names(df) && ("predicted" in names(df) || "ground_truth" in names(df))
            println("✅ Has correct columns for submission")
        else
            push!(issues, "Submission file has wrong columns: $(names(df))")
        end
    catch
        push!(issues, "Cannot read submission file")
    end
end

# Print issues
if isempty(issues)
    println("✅ No major issues detected!")
else
    println("Found $(length(issues)) potential issues:")
    for (i, issue) in enumerate(issues)
        println("  $i. $issue")
    end
end

println()

# ============================================================================
# 6. RECOMMENDATIONS
# ============================================================================
println("💡 STEP 6: RECOMMENDATIONS FOR HACKATHON TEAMS")
println("-"^50)

recommendations = []

# Recommendation based on feature files
if length(feature_files) > 1
    push!(recommendations, "Clean up duplicate feature files - keep only:")
    push!(recommendations, "  • features_train.csv (for training)")
    push!(recommendations, "  • features_test.csv (for prediction)")
end

# Check if we have the full pipeline
has_extractor = "extract_features.jl" in files
has_trainer = "train_xgboost.jl" in files || "train_final.jl" in files
has_predictor = "predict.jl" in files || "make_predictions.jl" in files

pipeline_status = []
if has_extractor
    push!(pipeline_status, "✅ Feature extraction script")
else
    push!(pipeline_status, "❌ Missing feature extraction script")
end

if has_trainer
    push!(pipeline_status, "✅ Training script")
else
    push!(pipeline_status, "❌ Missing training script")
end

if has_predictor
    push!(pipeline_status, "✅ Prediction script")
else
    push!(pipeline_status, "❌ Missing prediction script")
end

println("ML PIPELINE STATUS:")
for status in pipeline_status
    println("  $status")
end

if has_extractor && has_trainer && has_predictor
    println("\n🎉 COMPLETE PIPELINE DETECTED!")
    println("Your team has all the necessary components!")
    
    push!(recommendations, "To run the complete pipeline:")
    push!(recommendations, "  1. julia extract_features.jl")
    push!(recommendations, "  2. julia train_xgboost.jl")
    push!(recommendations, "  3. julia predict.jl")
else
    missing_parts = []
    if !has_extractor
        push!(missing_parts, "feature extraction")
    end
    if !has_trainer
        push!(missing_parts, "model training")
    end
    if !has_predictor
        push!(missing_parts, "prediction")
    end
    
    println("\n⚠️  INCOMPLETE PIPELINE")
    println("Missing: $(join(missing_parts, ", "))")
    
    push!(recommendations, "Develop the missing parts: $(join(missing_parts, ", "))")
end

# Check for verification scripts
verification_scripts = filter(x -> occursin("verify", lowercase(x)) || 
                                   occursin("check", lowercase(x)), categories["Julia Scripts"])
if !isempty(verification_scripts)
    println("\n🔍 VERIFICATION SCRIPTS FOUND:")
    for script in verification_scripts
        println("  • $script")
    end
    push!(recommendations, "Run verification scripts to check data integrity")
else
    push!(recommendations, "Add verification scripts to check data quality")
end

# Print recommendations
if !isempty(recommendations)
    println("\n📋 RECOMMENDATIONS:")
    for (i, rec) in enumerate(recommendations)
        println("  $i. $rec")
    end
end

println()
println("="^80)
println("🔍 ANALYSIS COMPLETE")
println("="^80)

# Summary
println("\n📊 SUMMARY FOR HACKATHON JUDGES:")
println("• CSV Files: $(length(categories["CSV Files"]))")
println("• Julia Scripts: $(length(categories["Julia Scripts"]))")
println("• Python Scripts: $(length(categories["Python Scripts"]))")
println("• Model Files: $(length(categories["Model Files"]))")
println("• Issues Found: $(length(issues))")
println("• Pipeline Status: $(has_extractor && has_trainer && has_predictor ? "Complete" : "Incomplete")")

if train_csv !== nothing && test_csv !== nothing
    try
        train_size = nrow(CSV.read(train_csv, DataFrame))
        test_size = nrow(CSV.read(test_csv, DataFrame))
        println("• Dataset Size: $train_size train, $test_size test")
    catch
        println("• Dataset Size: Could not determine")
    end
end

println("="^80)
