#!/usr/bin/env perl
use strict;
use warnings;
use Term::ANSIColor;

# 基本設定
my $command     = "python3 ./deep-photo-styletransfer-tf/deep_photostyle.py ";
my $content     = "--content_image_path " . $ENV{CONTENT};
my $style       = "--style_image_path " . $ENV{STYLE};
my $content_seg = "--content_seg_path " . $ENV{CONTENTSEG};
my $style_seg   = "--style_seg_path " . $ENV{STYLESEG};
my $style_option= "--style_option 2";
my $output      = "--output_image " . $ENV{OUTPUT};

open(DATAFILE, "< pair.txt") or die("error :$!");

while (my $line = <DATAFILE>){
    if ($line =~ /^\#/) {
	next;
    }
    my $cnt_file = "";
    my $sty_file = "";
    my $cnt_seg_file = "";
    my $sty_seg_file = "";
    my $out_file = "";
    chomp($line);
    if ($line =~ /(.+?)\s(.+?)\s(.+?)\s(.+?)$/){
	$cnt_file = $1;
	$cnt_seg_file = $2;
	$sty_file = $3;
	$sty_seg_file = $4;
    }

    $out_file = $cnt_file . $sty_file;

    logger("Start content: " . $cnt_file . " seg: ".  $cnt_seg_file . " style: " . $sty_file . " seg: " . $sty_seg_file);

    system($command . " " . $content . $cnt_file . " " . $style. $sty_file . " " . $content_seg . $cnt_seg_file . " " . $style_seg . $sty_seg_file . " " . $output . $out_file . " " . $style_option);

    logger("End output " . $out_file);
}

# Logger
sub logger {
    my ($message) = @_;
    print color('bold red') . "[LOG]:" . $message . "\n";
    print color('reset');
}
