#!/usr/bin/env perl
use strict;
use warnings;

# 基本設定
my $command     = "python3 ./neural-style/neural_style.py ";
my $content     = "--content " . $ENV{CONTENT};
my $style       = "--styles " . $ENV{STYLE};
my $output      = "--output " . $ENV{OUTPUT};

open(DATAFILE, "< pair.txt") or die("error :$!");

while (my $line = <DATAFILE>){
    if ($line =~ /^\#/) {
	next;
    }
    my $sty_file = "";
    my $cnt_file = "";
    my $out_file = "";
    chomp($line);
    if ($line =~ /(.+?)\s(.+?)$/){
	$cnt_file = $1;
	$sty_file = $2;
    }
    $out_file = $cnt_file . $sty_file;

    logger("Start content: " . $cnt_file . " style: " . $sty_file;

    system($command . " " . $content . $cnt_file . " " . $style. $sty_file . " " . $output . $out_file);

    logger("End output " . $output_file;
}

# Logger
sub logger {
    my ($message) = @_;
    print color('bold red') . "[LOG]:" . $message . "\n";
}
