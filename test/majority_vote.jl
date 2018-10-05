using Test
using kNN

@testset "MajorityVote" begin
	votes = ["A", "B", "B", "A", "C", "B"]
    @test isequal(kNN.majority_vote(votes), "B")

    ["A", "B", "B", "A", "C", "B", "C", "C"]
    @test isequal(kNN.majority_vote(votes), "B")

    votes = ["A", "B", "B", "A", "C", "B", "C", "C", "C"]
    @test isequal(kNN.majority_vote(votes), "C")
end
