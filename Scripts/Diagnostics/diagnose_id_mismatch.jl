# clear_diagnostic.jl
using DataFrames, CSV

println("🔍 CLEAR DIAGNOSTIC")
println("="^50)

# 1. Get actual folder contents
println("\n📂 ACTUAL FOLDER CONTENTS:")
train_files = readdir("train/")
test_files = readdir("test/")

println("First 10 train files: $(train_files[1:10])")
println("First 10 test files: $(test_files[1:10])")

# 2. Get CSV contents
println("\n📄 CSV FILE CONTENTS:")
train_csv = CSV.read("train.csv", DataFrame)
test_csv = CSV.read("test.csv", DataFrame)

println("First 5 train CSV IDs: $(first(train_csv.id, 5))")
println("First 5 test CSV IDs: $(first(test_csv.id, 5))")

# 3. Get feature file contents
println("\n🎯 FEATURE FILE CONTENTS:")
if isfile("features_train.csv")
    feat_train = CSV.read("features_train.csv", DataFrame)
    println("First column name: $(names(feat_train)[1])")
    println("First 5 IDs: $(first(feat_train[:, 1], 5))")
end

# 4. SIMPLE COMPARISON
println("\n🔍 SIMPLE COMPARISON:")
println("-"^30)

# Convert CSV IDs without .png
csv_train_ids = [replace(id, ".png" => "") for id in train_csv.id]
csv_test_ids = [replace(id, ".png" => "") for id in test_csv.id]

# Convert folder IDs without .png
folder_train_ids = [replace(f, ".png" => "") for f in train_files if endswith(f, ".png")]
folder_test_ids = [replace(f, ".png" => "") for f in test_files if endswith(f, ".png")]

println("Comparing first 5 IDs:")
println("CSV Train: $(csv_train_ids[1:5])")
println("Folder Train: $(folder_train_ids[1:5])")
println("")
println("CSV Test: $(csv_test_ids[1:5])")
println("Folder Test: $(folder_test_ids[1:5])")

# 5. Check if first ID from CSV exists in folder
println("\n🔎 QUICK EXISTENCE CHECK:")
first_csv_train_id = csv_train_ids[1]
first_csv_test_id = csv_test_ids[1]

println("Does CSV train ID '$first_csv_train_id' exist in train folder?")
println("  $(first_csv_train_id in folder_train_ids ? "✅ YES" : "❌ NO")")

println("Does CSV test ID '$first_csv_test_id' exist in test folder?")
println("  $(first_csv_test_id in folder_test_ids ? "✅ YES" : "❌ NO")")

# 6. Show sample of what actually exists
println("\n📊 SAMPLE OF ACTUAL DATA:")
println("Train folder sample IDs: $(folder_train_ids[1:20])")
println("Test folder sample IDs: $(folder_test_ids[1:20])")
