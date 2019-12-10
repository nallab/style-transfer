#!/usr/bin/env perl
use strict;
use warnings;
use Term::ANSIColor;
use File::Copy 'move';

# command line argument
my $count = 1;
my $dir = "testdata";

logger("To separate test data from datasets.");
my @all_files = glob "*";

mkdir $dir or die "$dir cannot create ... ";


while (my $file = <@all_files>) {
    $count--;
    if ($count == 0) {
	move $file, $dir;
	logger("Move $file to $dir");
	$count = 20;
    }
}

# Logger
sub logger {
    my ($message) = @_;
    print color('bold red') . "[LOG]:" . $message . "\n";
    print color('reset');
}
